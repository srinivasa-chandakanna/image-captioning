# imgcapgen/models/resnet_finetune2_attention_LSTM.py
"""
ResNet-50 (partially fine-tuned on layer2, layer3, layer4) + LSTM with Bahdanau Attention image captioning model.

Encoder: ResNet-50 pretrained, unfreezes layer2-4, outputs spatial feature grid (7x7) for attention.
Decoder: Embedding + Bahdanau Attention over image regions + LSTM + FCN for vocab prediction.
"""

import torch
import torch.nn as nn
import torchvision.models as models

# ------------------------------
# CNN Encoder (ResNet-50)
# ------------------------------
class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)

        # Freeze all layers, then unfreeze layer2-4 for finetuning
        for param in resnet.parameters():
            param.requires_grad = False
        for name, param in resnet.named_parameters():
            if "layer2" in name or "layer3" in name or "layer4" in name:
                param.requires_grad = True

        modules = list(resnet.children())[:-2]  # keep conv up to last layer
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)   # [B, 2048, 7, 7]
        features = features.view(features.size(0), features.size(1), -1)  # [B, 2048, 49]
        features = features.permute(0, 2, 1)  # [B, 49, 2048]
        return features

# ------------------------------
# Attention module
# ------------------------------
class BahdanauAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, attn_dim):
        super(BahdanauAttention, self).__init__()
        self.W = nn.Linear(feature_dim, attn_dim)
        self.U = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1)

    def forward(self, features, hidden):
        # features: [B, 49, 2048]
        # hidden:   [B, hidden_dim]
        hidden_exp = hidden.unsqueeze(1)  # [B, 1, hidden_dim]
        score = torch.tanh(self.W(features) + self.U(hidden_exp))  # [B, 49, attn_dim]
        attention = self.v(score).squeeze(-1)  # [B, 49]
        alpha = torch.softmax(attention, dim=1)
        context = (features * alpha.unsqueeze(-1)).sum(dim=1)  # [B, 2048]
        return context, alpha

# ------------------------------
# Decoder with Attention & LSTM
# ------------------------------
class DecoderWithAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, feature_dim=2048, attn_dim=256, num_layers=1, drop_prob=0.3):
        super(DecoderWithAttention, self).__init__()
        self.attention = BahdanauAttention(feature_dim, hidden_size, attn_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + feature_dim, hidden_size)
        self.fcn = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(drop_prob)
        self.hidden_size = hidden_size

    def forward(self, features, captions):
        batch_size = features.size(0)
        hidden = torch.zeros(batch_size, self.hidden_size).to(features.device)
        cell = torch.zeros(batch_size, self.hidden_size).to(features.device)

        embeddings = self.embedding(captions[:, :-1])  # [B, seq_len-1, embed]
        outputs = []

        for t in range(embeddings.size(1)):
            context, alpha = self.attention(features, hidden)  # [B, 2048]
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=-1)
            hidden, cell = self.lstm(lstm_input, (hidden, cell))
            output = self.fcn(self.dropout(hidden))
            outputs.append(output.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # [B, seq_len-1, vocab_size]
        return outputs

    def generate_caption(self, features, max_len=20, vocab=None):
        batch_size = features.size(0)
        hidden = torch.zeros(batch_size, self.hidden_size).to(features.device)
        cell = torch.zeros(batch_size, self.hidden_size).to(features.device)

        inputs = torch.tensor([vocab.stoi["<SOS>"]] * batch_size).to(features.device)
        inputs = self.embedding(inputs)  # [B, embed]

        captions = []
        for _ in range(max_len):
            context, alpha = self.attention(features, hidden)
            lstm_input = torch.cat([inputs, context], dim=-1)
            hidden, cell = self.lstm(lstm_input, (hidden, cell))
            output = self.fcn(hidden)
            predicted = output.argmax(dim=1)
            captions.append(predicted.cpu().numpy())
            inputs = self.embedding(predicted)

        captions = list(zip(*captions))  # transpose from time-major to batch-major
        decoded = [[vocab.itos[idx] for idx in cap] for cap in captions]
        # ✅ NEW: return simpler for batch=1
        if batch_size == 1:
            return decoded[0]
        return decoded
# ------------------------------
# Combined Encoder-Decoder module
# ------------------------------
class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attn_dim=256, num_layers=1, drop_prob=0.3):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderWithAttention(embed_size, hidden_size, vocab_size, attn_dim=attn_dim, num_layers=num_layers, drop_prob=drop_prob)

    def forward(self, images, captions):
        features = self.encoder(images)  # [B, 49, 2048]
        outputs = self.decoder(features, captions)
        return outputs
