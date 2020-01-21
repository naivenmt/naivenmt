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

# Attention mechanisms implementation.
# Attention machenism : https://blog.floydhub.com/attention-mechanism/

import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    """Bahdanau (additive-stype) attention. Bahdanau attention use the decoder output of the previous time step."""

    def __init__(self, units):
        """Init.

        Args:
            units: Python integer, hidden size of attention
        """
        super(BahdanauAttention, self).__init__(name='BahdanauAttention')
        self.wq = tf.keras.layers.Dense(units)
        self.wv = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_state, enc_output):
        """Forward pass one step in decoder.

        Args:
            dec_state: Tensor, decoder's state, shape (batch_size, units)
            enc_output: Tensor, encoder's output, shape (batch_size, src_seq_len, units)

        Returns:
            context: Tensor, shape (batch_size, units)
            attn_weights: Tensor, shape (batch_size, src_seq_len)
        """
        batch_size, src_seq_len = tf.shape(enc_output)[0], tf.shape(enc_output)[1]
        dec_state = tf.expand_dims(dec_state, 1)  # shape (batch_size, 1, units)
        score = self.V(tf.nn.tanh(self.wq(dec_state) + self.wv(enc_output)))  # shape (batch_size, src_seq_len, 1)
        attn_weights = tf.nn.softmax(score, axis=1)  # shape (batch_size, src_seq_len, 1)
        context = tf.reduce_sum(attn_weights * enc_output, axis=1)  # shape (batch_size, units)
        attn_weights = tf.reshape(attn_weights, shape=(batch_size, src_seq_len))  # shape (batch_size, src_seq_len)
        return context, attn_weights


class LuongAttention(tf.keras.layers.Layer):
    """Luong (multiplicative) attention. Luong attention use the current decoder output to compute attention weights."""

    def __init__(self, units):
        super(LuongAttention, self).__init__(name='LuongAttention')
        self.w = tf.keras.layers.Dense(units)

    def call(self, dec_output, enc_output):
        """Forward pass one time step in decoder.

        Args:
            dec_output: Tensor, decoder's output, shape (batch_size, 1, units)
            enc_output: Tensor, encoder's output, shape (batch_size, src_seq_len, units)

        Returns:
            context: Tensor, shape (batch_size, units)
            attn_weights: Tensor, shape (batch_size, src_seq_len)
        """
        batch_size, src_seq_len = tf.shape(enc_output)[0], tf.shape(enc_output)[1]
        dec_output = self.w(dec_output)  # shape (batch_size, 1, units)
        score = tf.matmul(enc_output, dec_output, transpose_b=True)  # shape (batch_size, src_seq_len, 1)
        attn_weights = tf.nn.softmax(score, axis=1)  # shape (batch_size, src_seq_len, 1)
        context = tf.reduce_sum(attn_weights * enc_output, axis=1)  # shape (batch_size, units)
        attn_weights = tf.reshape(attn_weights, shape=(batch_size, src_seq_len))  # shape (batch_size, src_seq_len)
        return context, attn_weights
