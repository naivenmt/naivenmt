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


def build_uni_rnn(unit_name, units):
    """Build uni-directional RNN."""
    valid_unit_names = ['lstm', 'gru']
    if unit_name.lower() not in valid_unit_names:
        raise ValueError('Invalid `unit_name`: %s, must be one of %s' % (unit_name, valid_unit_names))
    if 'lstm' == unit_name.lower():
        return tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
    elif 'gru' == unit_name.lower():
        return tf.keras.layers.GRU(units, return_sequences=True, return_state=True)
    else:
        raise ValueError('Invalid `unit_name`: %s, must be one of %s' % (unit_name, valid_unit_names))


def build_bi_rnn(unit_name, units, merge_mode='concat'):
    """Build bi-directional RNN."""
    valid_unit_names = ['lstm', 'gru']
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
    else:
        raise ValueError('Invalid `unit_name`: %s, must be one of %s' % (unit_name, valid_unit_names))


def build_embedding(vocab_size, embedding_size, embedding=None):
    """Build new embedding layer or use the exist one.

    Args:
        vocab_size: Python integer, number of vocabs
        embedding_size: Python integer, embedding dimession
        embedding: Tensor, instance of `tf.keras.layers.Embedding`

    Returns:
        Embedding layer.
    """
    if embedding is not None:
        return embedding
    if vocab_size is None or embedding_size is None:
        raise ValueError('Both `vocab_size` and `embedding_size` can not be None if `embedding` is None.')
    return tf.keras.layers.Embedding(vocab_size, embedding_size)
