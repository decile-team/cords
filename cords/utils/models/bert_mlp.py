from transformers import BertModel, BertConfig
import torch.nn as nn
import torch


class BERTMLPModel(nn.Module):
    def __init__(self, config, checkpoint="bert-base-uncased"):
        super(BERTMLPModel, self).__init__()
        self.config = config
        self.checkpoint = checkpoint
        self.bert = BertModel.from_pretrained(checkpoint)
        classifier_dropout = (
        config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        ### New layers:
        self.linear1 = nn.Linear(768, config.l1)
        self.linear2 = nn.Linear(config.l1, config.num_classes)

    def get_embedding_dim(self):
        return self.config.l1

    def forward(self, input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        finetune=True, freeze=False, last=False):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if freeze:
            with torch.no_grad():
                outputs = self.bert(input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask,
                                    inputs_embeds=inputs_embeds,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict,)
                # sequence_output has the following shape: (batch_size, sequence_length, 768)
                pooled_output = self.dropout(outputs[1])
                linear1_output = self.linear1(pooled_output) ## extract the 1st token's embeddings              
        elif finetune:
                outputs = self.bert(input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask,
                                    inputs_embeds=inputs_embeds,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict,)
                pooled_output = self.dropout(outputs[1])
                # sequence_output has the following shape: (batch_size, sequence_length, 768)
                linear1_output = self.linear1(pooled_output) ## extract the 1st token's embeddings
        else:
            with torch.no_grad():
                outputs = self.bert(input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask,
                                    inputs_embeds=inputs_embeds,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict,)
                pooled_output = self.dropout(outputs[1])
            # sequence_output has the following shape: (batch_size, sequence_length, 768)
            linear1_output = self.linear1(pooled_output) ## extract the 1st token's embeddings
        linear2_output = self.linear2(linear1_output)
        if last:
            return linear2_output, linear1_output
        else:
            return linear2_output

if __name__ == "__main__":
    configuration = BertConfig()
    setattr(configuration, 'l1', 512)
    setattr(configuration, 'num_classes', 2)
    model = BERTMLPModel(configuration)
    print(model)