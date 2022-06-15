from blocks import * 

"""
Return 
final_layer: bz, tar_seg_len, target_vocab_size
"""
class Transformer(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()

        self.enc = Encoder(n_layers=n_layers, d_model=d_model, n_heads=n_heads,
        dff=dff, input_vocab_size=input_vocab_size)

        self.dec = Decoder(n_layers=n_layers, d_model=d_model, n_heads=n_heads,
        dff=dff, target_vocab_size=target_vocab_size)

        self.fc = nn.Linear(d_model, target_vocab_size)

    def create_msks(self, inp, tar):
        
        padding_msk = create_padding_mask(inp)

        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask         = create_look_ahead_mask(tar.shape[1])

        # only select masking position
        look_ahead_mask = torch.max(dec_target_padding_mask, look_ahead_mask)

        return padding_msk, look_ahead_mask
        
    def forward(self, inpts):
        inp, tar = inpts
        padding_msk, look_ahead_mask = self.create_msks(inp, tar)
        enc_output = self.enc(inp, padding_msk)
        dec_output, attn_wts = self.dec(tar, enc_output, look_ahead_mask, padding_msk)

        final_layer = self.fc(dec_output)   
        
        return final_layer, attn_wts

sample_transformer = Transformer(
    n_layers=6, d_model=512, n_heads=8, dff=2048,
    input_vocab_size=8500, target_vocab_size=8000)

temp_input = torch.randint(0, 200, (64, 38)) 
temp_target = torch.randint(0, 200, (64, 36)) 

fn_out, att_wts = sample_transformer([temp_input, temp_target])
print('fn_out:', fn_out.shape)  

"""
Machine Translation with Transformer
Input: n_layers=6, d_model=512, n_heads=8, dff=2048,
        input_vocab_size=8500, target_vocab_size=8000
Return: torch.Size([64, 36, 8000]) #(batch_size, tar_seq_len, target_vocab_size)
"""
