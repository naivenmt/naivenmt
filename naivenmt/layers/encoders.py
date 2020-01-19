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


def _build_uni_rnn(unit_name, units):
    valid_unit_names = ['lstm', 'gru', 'rnn']
    if unit_name.lower() not in valid_unit_names:
        raise ValueError('Invalid `unit_name`: %s, must be one of %s' % (unit_name, valid_unit_names))
    if 'lstm' == unit_name.lower():
        return tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
    elif 'gru' == unit_name.lower():
        return tf.keras.layers.GRU(units, return_sequences=True, return_state=True)
    elif 'rnn' == unit_name.lower():
        return tf.keras.layers.RNN(units, return_sequences=True, return_state=True)
    else:
        raise ValueError('Invalid `unit_name`: %s, must be one of %s' % (unit_name, valid_unit_names))


def _build_bi_rnn(unit_name, units, merge_mode='add'):
    valid_unit_names = ['lstm', 'gru', 'rnn']
    if unit_name.lower() not in valid_unit_names:
        raise ValueError('Invalid `unit_name`: %s, must be one of %s' % (unit_name, valid_unit_names))
    if 'lstm' == unit_name.lower():
        rnn = tf.keras.layers.Bidirectional(
            layer=tf.keras.layers.LSTM(units, return_sequences=True, return_state=True, go_backwards=False),
            backward_layer=tf.keras.layers.LSTM(units, return_sequences=True, return_state=True, go_backwards=True),
            merge_mode=merge_mode)
        return rnn
    elif 'gru' == unit_name.lower():
        rnn = tf.keras.layers.Bidirectional(
            layer=tf.keras.layers.GRU(units, return_sequences=True, return_state=True, go_backwards=False),
            backward_layer=tf.keras.layers.GRU(units, return_sequences=True, return_state=True, go_backwards=True),
            merge_mode=merge_mode)
        return rnn
    elif 'rnn' == unit_name.lower():
        rnn = tf.keras.layers.Bidirectional(
            layer=tf.keras.layers.RNN(units, return_sequences=True, return_state=True, go_backwards=False),
            backward_layer=tf.keras.layers.RNN(units, return_sequences=true, return_state=True, go_backwards=True),
            merge_mode=merge_mode)
        return rnn
    else:
        raise ValueError('Invalid `unit_name`: %s, must be one of %s' % (unit_name, valid_unit_names))


def _build_embedding(vocab_size, embedding_size, embedding):
    if embedding is not None:
        return embedding
    if vocab_size is None or embedding_size is None:
        raise ValueError('Both `vocab_size` and `embedding_size` can not be None if `embedding` is None.')
    return tf.keras.layers.Embedding(vocab_size, embedding_size)


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
        self.rnn = _build_uni_rnn(unit_name, units)
        self.embedding = _build_embedding(vocab_size, embedding_size, embedding)

    def call(self, inputs, training=None, mask=None, initial_state=None):
        """Forward pass.

        Args:
            inputs: Tensor, the source sequence, shape (batch_size, src_seq_len)
            training: Python boolean, is training or not.
            mask: Tensor

        Returns:
            output: Tensor, encoder's output, shape (batch_size, src_seq_len, units)
            state: Tensor, encoder's state, shape (batch_size, units)
        """
        x = inputs
        x = self.embedding(x)
        output, state = self.rnn(x, initial_state=initial_state)
        # output, state = outputs[0], outputs[1]
        return output, state

    def create_initial_state(self, x):
        batch_szie = tf.shape(x)[0]
        return tf.zeros((batch_szie, self.units))


class BiRNNEncoder(tf.keras.Model):

    def __init__(self, units, vocab_size=None, embedding_size=None, embedding=None, unit_name='lstm', merge_mode='add'):
        """Init.

        Args:
            units: Python integer, hidden units of RNN.
            vocab_size: (Optional) Python integer, number of vocabs of source sequence. If None, `embedding` must not be None.
            embedding_size: (Optional) Python integer, dimenssion of embedding. If None, `embedding` must not be None.
            embedding: (Optional) Instance of tf.keras.layers.Embedding. If None, create new embedding layer, 
                so `vocab_size` and `embedding_size` could not be None.
            unit_name: Python string, the name of RNN.
        """
        super(BiRNNEncoder, self).__init__(name='BiRNNEncoder')
        self.units = units
        self.rnn = _build_bi_rnn(unit_name, units, merge_mode)
        self.embedding = _build_embedding(vocab_size, embedding_size, embedding)

    def call(self, inputs, training=None, mask=None, initial_state=None):
        """Forward pass.

        Args:
            inputs: Tensor, the source sequence, shape (batch_size, src_seq_len)
            training: Python boolean, is training or not.
            mask: Tensor

        Returns:
            output: Tensor, encoder's output, shape (batch_size, src_seq_len, units). If merge_mode is `concat`,
                the shape is (batch_size, src_seql_len, units)
            state: Tensor, encoder's state, shape (batch_size, units). If merge_mode is `concat`,
                the shape is (batch_size, src_seq_len, 2 * units)
        """
        x = inputs
        x = self.embedding(x)
        outputs = self.rnn(x, initial_state=initial_state)
        output, state = outputs[0], outputs[1]
        return output, state

    def create_initial_state(self, x):
        batch_szie = tf.shape(x)[0]
        state = tf.zeros((batch_szie, self.units))
        return [state, state]
