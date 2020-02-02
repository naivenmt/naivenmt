# Copyright 2020 The naivenmt authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from naivenmt import utils
from naivenmt.layers.attentions import BahdanauAttention, LuongAttention


def _build_attention(units, attention):
    if isinstance(attention, str):
        if 'bahdanau' == attention.lower():
            return BahdanauAttention(units)
        elif 'luong' == attention.lower():
            return LuongAttention(units)
        else:
            raise ValueError('Invalid attention name: %s, must bet one of [`bahdanau`, `luong`].' % attention)
    elif isinstance(attention, BahdanauAttention):
        return BahdanauAttention(units)
    elif isinstance(attention, LuongAttention):
        return LuongAttention(units)
    else:
        raise ValueError("""Invalid argument `attention`. 
                Must be a name of attention mechanism in [`bahdanau`, `luong`],
                or an instance of attention mechanism in [`BahdanauAttention`, `LuongAttention`]""")


class UniRNNDecoder(tf.keras.Model):
    """Uni-directional RNN decoder."""

    def __init__(self, units, vocab_size, embedding_size=None, embedding=None, unit_name='gru', attention='bahdanau'):
        super(UniRNNDecoder, self).__init__(name='UniRNNDecoder')
        self.units = units
        self.vocab_size = vocab_size
        self.unit_name = unit_name.lower()
        self.embedding = utils.build_embedding(vocab_size, embedding_size, embedding)
        self.rnn = utils.build_uni_rnn(self.unit_name, self.units)
        self.attention = _build_attention(units, attention)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, state, enc_output):
        """Forward pass one time step.

        Args:
            x: Tensor, decoder's input sequence, shape (batch_size, 1)
            state: Tensor, decoder's state, shape (batch_size, units)
            enc_output: Tensor, encoder's output, shape (batch_size, src_seq_len, units)

        Returns:
            output: Tensor, next word prob distribution over target vocab, shape (batch, tgt_vocab_size)
            state: Tensor, decoder's state for next time step, shape (batch_size, units)
            attn_weights: Tensor, attention weights over src sequence in this time step, shape (batch_size, src_seq_len)
        """
        if isinstance(self.attention, BahdanauAttention):
            return self._bahdanau_attention_decode(x, state, enc_output)
        elif isinstance(self.attention, LuongAttention):
            return self._luong_attention_decode(x, state, enc_output)
        else:
            raise ValueError(
                'Invalid attention mechanism. Must be an instance of [`BahdanauAttention`, `LuongAttention`]')

    def _bahdanau_attention_decode(self, x, state, enc_output):
        """Bahdanau attention use the decoder output of the previous time step.

        Args:
            x: Tensor, decoder's input sequence, shape (batch_size, 1)
            state: Tensor, decoder's state, shape (batch_size, units)
            enc_output: Tensor, encoder's output, shape (batch_size, src_seq_len, units)

        Returns:
            output: Tensor, next word prob distribution over target vocab, shape (batch_size, tgt_vocab_size)
            state: Tensor, decoder's state for next time step, shape (batch_size, units).
                If RNN is LSTM, returns (state_h, state_c).
            attn_weights: Tensor, attention weights over src sequence in this time step, shape (batch_size, src_seq_len)
        """
        # only take state_h from lstm
        rnn_state = state[0] if 'lstm' == self.unit_name else state
        context, attn_weights = self.attention(rnn_state, enc_output)
        x = self.embedding(x)  # shape (batch_size, src_seq_len, embedding_size)
        # shape (batch_size, src_seq_len, embedding_size + units)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        if 'lstm' == self.unit_name:
            # output shape (batch_size, 1, units)
            output, state_h, state_c = self.rnn(x, initial_state=state)
            state = (state_h, state_c)
        else:
            output, state = self.rnn(x, initial_state=state)
        output = tf.reshape(output, (-1, output.shape[-1]))  # shape (batch_size, units)
        output = self.fc(output)  # shape (batch_size, vocab_size)
        return output, state, attn_weights

    def _luong_attention_decode(self, x, state, enc_output):
        """Luong attention use current decoder output to compute alignment score.
            Pytorch tutorial: https://blog.floydhub.com/attention-mechanism/

        Args:
            x: Tensor, decoder's input sequence, shape (batch_size, 1)
            state: Tensor, decoder's state, shape (batch_size, units)
            enc_output: Tensor, encoder's output, shape (batch_size, src_seq_len, units)

        Returns:
            output: Tensor, next word prob distribution over target vocab, shape (batch, tgt_vocab_size)
            state: Tensor, decoder's state for next time step, shape (batch_size, units).
                If RNN is LSTM, returns (state_h, state_c).
            attn_weights: Tensor, attention weights over src sequence in this time step, shape (batch_size, src_seq_len)
        """
        x = self.embedding(x)  # shape (batch_size, tgt_seq_len, embedding_size), tgt_seq_len == 1.
        if 'lstm' == self.unit_name:
            dec_output, dec_state_h, dec_state_c = self.rnn(x, initial_state=state)
            dec_state = (dec_state_h, dec_state_c)
        else:
            dec_output, dec_state = self.rnn(x, initial_state=state)
        context, attn_weights = self.attention(dec_output, enc_output)
        dec_output = tf.reshape(dec_output, (-1, dec_output.shape[-1]))  # shape (batch_size, units)
        dec_output = self.fc(dec_output)  # shape (batch_size, tgt_vocab_size)
        return dec_output, dec_state, attn_weights


class BiRNNDecoder(tf.keras.Model):

    def __init__(self,
                 units,
                 vocab_size,
                 embedding_size=None,
                 embedding=None,
                 unit_name='gru',
                 merge_mode='concat',
                 attention='bahdanau'):
        super(BiRNNDecoder, self).__init__(name='BiRNNDecoder')
        self.units = units
        self.vocab_size = vocab_size
        self.unit_name = unit_name.lower()
        self.merge_mode = merge_mode.lower()
        self.embedding = utils.build_embedding(vocab_size, embedding_size, embedding)
        self.rnn = utils.build_bi_rnn(self.unit_name, units, self.merge_mode)
        self.attention = _build_attention(units, attention)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, state, enc_output):
        """Forward pass one time step.

        Args:
            x: Tensor, decoder's input sequence, shape (batch_size, 1)
            state: Tensor, decoder's state, shape (batch_size, units)
            enc_output: Tensor, encoder's output, shape (batch_size, src_seq_len, units)

        Returns:
            output: Tensor, next word prob distribution over target vocab, shape (batch, tgt_vocab_size)
            state: Tensor, decoder's state for next time step, shape (batch_size, units)
            attn_weights: Tensor, attention weights over src sequence in this time step, shape (batch_size, src_seq_len)
        """
        if isinstance(self.attention, BahdanauAttention):
            return self._bahdanau_attention_decode(x, state, enc_output)
        elif isinstance(self.attention, LuongAttention):
            return self._luong_attention_decode(x, state, enc_output)
        else:
            raise ValueError(
                'Invalid attention mechanism. Must be an instance of [`BahdanauAttention`, `LuongAttention`]')

    def _bahdanau_attention_decode(self, x, state, enc_output):
        """Bahdanau attention use the decoder output of the previous time step.

        Args:
            x: Tensor, decoder's input sequence, shape (batch_size, 1)
            state: Tensor, decoder's state, shape (batch_size, units)
            enc_output: Tensor, encoder's output, shape (batch_size, src_seq_len, units)

        Returns:
            output: Tensor, next word prob distribution over target vocab, shape (batch_size, tgt_vocab_size)
            state: Tensor, decoder's state for next time step, shape (batch_size, units).
                If RNN is LSTM, returns (state_h, state_c).
            attn_weights: Tensor, attention weights over src sequence in this time step, shape (batch_size, src_seq_len)
        """
        # only take state_h from lstm
        if 'lstm' == self.unit_name:
            state_fw_h, _, state_bw_h, _ = state
            attn_state = tf.concat([state_fw_h, state_bw_h], axis=-1)
        else:
            state_fw, state_bw = state
            attn_state = tf.concat([state_fw, state_bw], axis=-1)
        context, attn_weights = self.attention(attn_state, enc_output)
        x = self.embedding(x)  # shape (batch_size, src_seq_len, embedding_size)
        # shape (batch_size, src_seq_len, embedding_size + units)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        if 'lstm' == self.unit_name:
            # output shape (batch_size, 1, units)
            # output, state = utils.bi_lstm(self.rnn, self.merge_mode, x, state)
            output, state_fw_h, state_fw_c, state_bw_h, state_bw_c = self.rnn(x, initial_state=state)
            state = (state_fw_h, state_fw_c, state_bw_h, state_bw_c)
        else:
            # output, state = utils.bi_gru(self.rnn, self.merge_mode, x, state)
            output, state_fw, state_bw = self.rnn(x, initial_state=state)
            state = (state_fw, state_bw)
        output = tf.reshape(output, (-1, output.shape[-1]))  # shape (batch_size, units)
        output = self.fc(output)  # shape (batch_size, vocab_size)
        return output, state, attn_weights

    def _luong_attention_decode(self, x, state, enc_output):
        """Luong attention use current decoder output to compute alignment score.
            Pytorch tutorial: https://blog.floydhub.com/attention-mechanism/

        Args:
            x: Tensor, decoder's input sequence, shape (batch_size, 1)
            state: Tensor, decoder's state, shape (batch_size, units)
            enc_output: Tensor, encoder's output, shape (batch_size, src_seq_len, units)

        Returns:
            output: Tensor, next word prob distribution over target vocab, shape (batch, tgt_vocab_size)
            state: Tensor, decoder's state for next time step, shape (batch_size, units).
                If RNN is LSTM, returns (state_h, state_c).
            attn_weights: Tensor, attention weights over src sequence in this time step, shape (batch_size, src_seq_len)
        """
        x = self.embedding(x)  # shape (batch_size, tgt_seq_len, embedding_size), tgt_seq_len == 1.
        if 'lstm' == self.unit_name:
            # dec_output, dec_state = utils.bi_lstm(self.rnn, self.merge_mode, x, state)
            output, state_fw_h, state_fw_c, state_bw_h, state_bw_c = self.rnn(x, initial_state=state)
            state = (state_fw_h, state_fw_c, state_bw_h, state_bw_c)
        else:
            # dec_output, dec_state = utils.bi_gru(self.rnn, self.merge_mode, x, state)
            output, state_fw, state_bw = self.rnn(x, initial_state=state)
            state = (state_fw, state_bw)
        context, attn_weights = self.attention(output, enc_output)
        output = tf.reshape(output, (-1, output.shape[-1]))  # shape (batch_size, units)
        output = self.fc(output)  # shape (batch_size, tgt_vocab_size)
        return output, state, attn_weights
