import torch
from torch import nn
from torch.nn import init
import torchvision
from torchvision import transforms
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LargeScaleEncoder(nn.Module):
    def __init__(self, pretrained=True, fine_tune=True):
        super(LargeScaleEncoder, self).__init__()

        self.fcn = torchvision.models.segmentation.fcn_resnet101(
            pretrained=pretrained)
        resnet = torchvision.models.resnet101(
            pretrained=True)  # pretrained ImageNet ResNet-101

        self.pool = nn.MaxPool2d(2, 2)

        if fine_tune:
            self.fine_tune()

    def forward(self, img):
        out = self.pool(img)
        out = self.fcn(out)['out']

        out = out.argmax(1)  # (batch_size, 128, 128)

        return out/1.  # (batch_size, 128, 128)

    def fine_tune(self):
        for p in self.fcn.parameters():
            p.require_grad = False

        layer = list(self.fcn.children())[0]
        for c in list(layer.children())[0:-2]:
            for p in c.parameters():
                p.require_grad = True


class SmallScaleEncoder(nn.Module):
    def __init__(self, pretrained=True, fine_tune=True):
        super(SmallScaleEncoder, self).__init__()
        resnet = torchvision.models.resnet101(
            pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.fine_tune(fine_tune=fine_tune)

    def forward(self, img):  # (batch_size, 3, 256, 256)
        output = self.resnet(img)
        output = output.permute(0, 2, 3, 1)
        return output  # (batch_size, 8, 8, 2048)

    def fine_tune(self, fine_tune):
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Encoder(nn.Module):
    def __init__(self, large_finetune=True, large_pretrained=True,
                 small_finetune=True, small_pretrained=True):
        super(Encoder, self).__init__()
        self.large_encoder = LargeScaleEncoder(
            large_pretrained, large_finetune)
        self.small_encoder = SmallScaleEncoder(
            small_pretrained, small_finetune)

    def forward(self, img):
        info = self.small_encoder(img)
        relation = self.large_encoder(img)

        return info, relation


class MSLSTMCell(nn.Module):
    def __init__(self, input_dim, h_dim):
        super(MSLSTMCell, self).__init__()

        self.W_i = nn.Linear(input_dim+h_dim, h_dim)
        self.W_o = nn.Linear(input_dim+h_dim, h_dim)
        self.W_f = nn.Linear(input_dim+h_dim, h_dim)
        self.W_g1 = nn.Linear(input_dim+h_dim, h_dim)
        self.W_g2 = nn.Linear(input_dim+h_dim, h_dim)
        self.reset_parameters(h_dim)

        self.tanh = nn.Tanh()
        self.sigma = nn.Sigmoid()
        self.mlp = nn.Linear(h_dim*2, h_dim)

    def forward(self, x, state):
        h, c, C = state

        x = torch.cat([x, h], dim=1)

        i = self.sigma(self.W_i(x))
        f = self.sigma(self.W_f(x))
        o = self.sigma(self.W_o(x))

        g1 = self.tanh(self.W_g1(x))
        g2 = self.tanh(self.W_g2(x))

        new_c = f*c+i*g1
        new_C = f*C+i*g2

        new_h = o*self.tanh(
            self.mlp(torch.cat([new_c, new_C], dim=1))
        )

        return new_h, new_c, new_C

    def reset_parameters(self, hidden_size):
        stdv = 1.0 / math.sqrt(hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        # linear layer to transform encoded image
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # linear layer to calculate values to be softmax-ed
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, feature, h):
        """
        Forward propagation.

        :param feature: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(
            feature)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(h)  # (batch_size, attention_dim)
        # (batch_size, num_pixels)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (
            feature * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class Decoder(nn.Module):
    def __init__(self, embed_dim, h_dim, vocab_size, attention_dim,
                 dropout=0.5,
                 info_shape=(None, 8, 8, 2048), relation_shape=(None, 128, 128)):

        super(Decoder, self).__init__()
        self.vocab_size = vocab_size

        self.info_num_pixel = info_shape[1]*info_shape[2]
        self.info_dim = info_shape[3]
        self.relation_num_pixel = relation_shape[1]*relation_shape[2]

        self.W_init_c = nn.Linear(self.info_dim, h_dim)
        self.W_init_h = nn.Linear(self.info_dim, h_dim)
        self.W_init_C = nn.Linear(self.relation_num_pixel, h_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(self.info_dim, h_dim, attention_dim)

        self.lstm = MSLSTMCell(embed_dim+self.info_dim, h_dim)

        self.f_beta = nn.Linear(h_dim, self.info_dim)
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(h_dim, vocab_size)
        self.dropout = nn.Dropout()

        self.init_weights()

    def forward(self, info, relation, captions, captions_lens):
        # (batch_size, 32*32,256)
        batch_size = captions.shape[0]

        info = info.view(batch_size, -1, self.info_dim)
        relation = relation.view(batch_size, -1)  # (batch_size, 128*128)

        captions_lens, sort_index = captions_lens.squeeze(
            1).sort(dim=0, descending=True)
        captions = captions[sort_index]
        info = info[sort_index]
        relation = relation[sort_index]

        sent_len = (captions_lens-1).tolist()
        words = self.embedding(captions)

        h, c, C = self._init_hidden_state(info, relation)

        predictions = torch.zeros(batch_size, max(
            sent_len), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(
            sent_len), self.info_num_pixel).to(device)

        for t in range(max(sent_len)):
            batch_size_t = sum([l > t for l in sent_len])
            word = words[:batch_size_t, t]
            info = info[:batch_size_t]
            h = h[:batch_size_t]
            c = c[:batch_size_t]
            C = C[:batch_size_t]

            pred, (h, c, C), alpha = self.next_pred(word, info, (h, c, C))

            predictions[:batch_size_t, t, :] = pred
            alphas[:batch_size_t, t, :] = alpha

        return predictions, alphas, sort_index

    def next_pred(self, word, info, state):
        h, c, C = state
        attention_weighted_feature, alpha = self.attention(info, h)
        # gating scalar, (batch_size_t, encoder_dim)
        gate = self.sigmoid(self.f_beta(h))
        attention_weighted_encoding = gate * attention_weighted_feature

        x = torch.cat((word, attention_weighted_feature), dim=1)
        h, c, C = self.lstm(x, (h, c, C))

        pred = self.fc(self.dropout(h))
        return pred, (h, c, C), alpha

    def _init_hidden_state(self, info_feature, relation_feature):
        info_feature = info_feature.mean(dim=1)

        h = self.W_init_h(info_feature)
        c = self.W_init_c(info_feature)
        C = self.W_init_C(relation_feature)

        return h, c, C

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)


if __name__ == "__main__":
    #img = torch.randn(32, 3, 256, 256).to(device)
    #encoder1 = LargeScaleEncoder(pretrained=False, fine_tune=True).to(device)
    #encoder2 = SmallScaleEncoder(pretrained=False, fine_tune=True).to(device)
    #a = encoder1(img)
    #b = encoder2(img)

    info = torch.randn(32, 32*32, 256).to(device)
    relation = torch.randn(32, 128*128).to(device)

    #from utils import CaptionDataset
    # train_loader = torch.utils.data.DataLoader(
    #CaptionDataset('../preprocessed_data', 'rsicd', 'TRAIN'),
    # batch_size=32, shuffle=True, pin_memory=True)
    #img, cap, caplen = next(iter(train_loader))
    #cap = cap.to(device)
    #caplen = caplen.to(device)
    decoder = Decoder(1024, 1024, 1000, 1024).to(device)
    #score, alphas, sort_ind = decoder(info, relation, cap, caplen)

    word = torch.randn(32, 1024).to(device)
    h, c, C = decoder._init_hidden_state(info, relation)
    decoder.next_pred(word, info, (h, c, C))
    decoder.next_pred(word[:5], info[:5], (h[:5], c[:5], C[:5]))