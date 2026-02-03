import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class PubMedBertTextEncoder(nn.Module):
    def __init__(self,
                 pretrained_model_name=r"BiomedNLPPubMedBERT/snapshots/d673b8835373c6fa116d6d8006b33d48734e305d"):
        super(PubMedBertTextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    def forward(self, texts):
        # texts: List[str]，如 ["patient has fever", "gene expression analysis"]
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(next(self.bert.parameters()).device)
        attention_mask = encoding['attention_mask'].to(next(self.bert.parameters()).device)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取[CLS]位置的特征向量
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
        return cls_embeddings


class PubMedTextEncoder(nn.Module):
    def __init__(self,
                 vocab_size=30522,
                 embed_dim=256,
                 nhead=8,
                 num_layers=3,
                 max_len=512,
                 tokenizer_path=r"BiomedNLPPubMedBERT/snapshots/d673b8835373c6fa116d6d8006b33d48734e305d"):
        super(PubMedTextEncoder, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # 2. 如果外部没传 vocab_size，自动从 tokenizer 获取
        if vocab_size is None:
            vocab_size = self.tokenizer.vocab_size

        # --- 调试代码：检查参数类型 ---
        if not isinstance(vocab_size, int):
            raise ValueError(f"vocab_size 必须是整数，但现在是: {type(vocab_size)}, 值: {vocab_size}")
        if not isinstance(embed_dim, int):
            raise ValueError(f"embed_dim 必须是整数，但现在是: {type(embed_dim)}, 值: {embed_dim}")
        # ---------------------------

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))

        # 3. Transformer 编码层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True,
            activation='gelu'  # 匹配 BERT 常用激活函数
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 线性层：映射到你需要的 768 维度
        self.fc = nn.Linear(embed_dim, 768)

        self.max_len = max_len

    def forward(self, texts):
        """
        输入: texts -> List[str], 例如 ["fever", "lung cancer"]
        输出: embeddings -> torch.Tensor (batch, 768)
        """
        # 获取模型所在的设备 (CPU/GPU)
        device = next(self.parameters()).device

        # 文本转 ID
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        ).to(device)

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # 生成 Embeddings 并加上位置编码
        x = self.embedding(input_ids)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]

        # 准备 Padding Mask (注意: PyTorch Transformer 需要的是布尔型 Mask，True 代表屏蔽)
        # padding_mask 维度: (batch, seq_len)
        src_key_padding_mask = (attention_mask == 0)

        # 通过 Transformer
        # out 维度: (batch, seq_len, embed_dim)
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # 池化处理：这里使用 Mean Pooling (平均池化) 提取整句特征
        # 我们只对非填充部分取平均值，这样效果更准
        mask_expanded = attention_mask.unsqueeze(-1).expand(out.size()).float()
        sum_embeddings = torch.sum(out * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # 映射到最终维度
        return self.fc(mean_pooled)


import pandas as pd


def read_texts_from_csv(csv_path, text_column='text'):
    """
    从CSV文件读取指定列的文本，返回文本列表
    :param csv_path: CSV文件路径
    :param text_column: 包含文本的列名
    :return: List[str]
    """
    df = pd.read_csv(csv_path)
    return df[text_column].tolist()


if __name__ == "__main__":
    encoder = PubMedBertTextEncoder(r"C:\Users\ta\Desktop\Swin-UNETR-main\BiomedNLPPubMedBERT\snapshots\d673b8835373c6fa116d6d8006b33d48734e305d")
    # 指定你的CSV文件路径和要读取的列名
    csv_path = r"G:\ToZhaoWenhao\patient_summary.csv"
    texts = "liver cancer for M"
    features = encoder(texts)
    print(f"Feature shape: {features.shape}")  # Expected to be (N, 768)

