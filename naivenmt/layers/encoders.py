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
import logging

import tensorflow as tf

from naivenmt import utils


class UniRNNEncoder(tf.keras.Model):

    def __init__(self, units, vocab_size=None, embedding_size=None, embedding=None, unit_name='lstm'):
        """Init.

        Args:
            units: Python integer, hidden units of RNN.
            vocab_size: (Optional) Python integer, number of vocabs of source sequence. If None, `embedding` must not be None.
            embedding_size: (Optional) Python integer, dimenssion of embedding. If None, `embedding` must not be None.
            embedding: (Optional) Instance of tf.keras.layers.Embedding. If None, create new embedding layer, 
                so `vocab_size` and `embedding_size` could not be None.
            unit_name: Python string, the name of RNN.
        """
        super(UniRNNEncoder, self).__init__(name='UniRNNEncoder')
        self.units = units
        self.unit_name = unit_name.lower()
        self.rnn = utils.build_uni_rnn(unit_name, units)
        self.embedding = utils.build_embedding(vocab_size, embedding_size, embedding)

    def call(self, inputs, training=None, mask=None):
        """Forward pass.

        Args:
            inputs: Tensor, the source sequence, shape (batch_size, src_seq_len)
            training: Python boolean, is training or not.
            mask: Tensor

        Returns:
            output: Tensor, encoder's output, shape (batch_size, src_seq_len, units)
            state: Tensor, encoder's state, shape (batch_size, units). 
                If merge mode is `concat`, shape is (batch_size, 2 * units).
                If RNN is a LSTM, returns tuple (state_h, state_c), each shape is (batch_size, units)
        """
        x = inputs
        x = self.embedding(x)
        if 'lstm' == self.unit_name:
            # output shape: (batch_size, src_seq_len), state_[c|h] shape: (batch_size, units)
            output, state_h, state_c = self.rnn(x)
            return output, (state_h, state_c)

        output, state = self.rnn(x)
        return output, state


class BiRNNEncoder(tf.keras.Model):

    def __init__(self, units, vocab_size=None, embedding_size=None, embedding=None, unit_name='lstm', merge_mode='sum'):
        """Init.

        Args:
            units: Python integer, hidden units of RNN.
            vocab_size: (Optional) Python integer, number of vocabs of source sequence. If None, `embedding` must not be None.
            embedding_size: (Optional) Python integer, dimenssion of embedding. If None, `embedding` must not be None.
            embedding: (Optional) Instance of tf.keras.layers.Embedding. If None, create new embedding layer, 
                so `vocab_size` and `embedding_size` could not be None.
            unit_name: Python string, the name of RNN.
            merge_mode: Python string, one of ['sum', 'ave', 'mul', 'concat']
        """
        super(BiRNNEncoder, self).__init__(name='BiRNNEncoder')
        self.units = units
        self.unit_name = unit_name.lower()
        self.merge_mode = merge_mode.lower()
        self.rnn = utils.build_bi_rnn(unit_name, units, merge_mode)
        self.embedding = utils.build_embedding(vocab_size, embedding_size, embedding)

    def call(self, inputs, training=None, mask=None, initial_state=None):
        """Forward pass.

        Args:
            inputs: Tensor, the source sequence, shape (batch_size, src_seq_len)
            training: Python boolean, is training or not.
            mask: Tensor

        Returns:
            output: Tensor, encoder's output, shape (batch_size, src_seq_len, units). 
                If merge_mode is `concat`, the shape is (batch_size, src_seql_len, 2 * units)
            state: Tensor, encoder's state, shape (batch_size, units). 
                If merge_mode is `concat`, the shape is (batch_size, 2 * units).
                If RNN is LSTM, returns tuple (state_h, state_c), each shape is (batch_size, units)
        """
        x = inputs
        x = self.embedding(x)

        if 'lstm' == self.unit_name:
            # bidirectional lstm does not merge state_h and state_c
            output, state_fw_h, state_fw_c, state_bw_h, state_bw_c = self.rnn(x)
            if 'ave' == self.merge_mode:
                state_h = (state_fw_h + state_bw_h) / 2.0
                state_c = (state_fw_c + state_bw_c) / 2.0
            elif 'sum' == self.merge_mode:
                state_h = (state_fw_h + state_bw_h)
                state_c = (state_fw_c + state_bw_c)
            elif 'mul' == self.merge_mode:
                state_h = state_fw_h * state_bw_h
                state_c = state_fw_c * state_bw_c
            elif 'concat' == self.merge_mode:
                state_h = tf.concat([state_fw_h, state_bw_h], axis=-1)
                state_c = tf.concat([state_fw_c, state_bw_c], axis=-1)
            else:
                state_h = tf.concat([state_fw_h, state_bw_h], axis=-1)
                state_c = tf.concat([state_fw_c, state_bw_c], axis=-1)
            return output, (state_h, state_c)
        # bidirectional gru does not merge state
        output, state_fw, state_bw = self.rnn(x)
        if 'ave' == self.merge_mode:
            state = (state_fw + state_bw)/2.0
        elif 'sum' == self.merge_mode:
            state = state_fw + state_bw
        elif 'mul' == self.merge_mode:
            state = state_fw * state_bw
        elif 'concat' == self.merge_mode:
            state = tf.concat([state_fw, state_bw], axis=-1)
        else:
            state = tf.concat([state_fw, state_bw], axis=-1)
        return output, state
