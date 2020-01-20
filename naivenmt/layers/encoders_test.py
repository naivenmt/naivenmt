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

from naivenmt.layers.encoders import BiRNNEncoder, UniRNNEncoder

UNITS = 16
VOCABS_ZIE = 10
EMBEDDING_SIZE = 32


class EncodersTest(tf.test.TestCase):

    def testUniGRUEncoder(self):
        x = tf.constant([1, 2, 3, 4, 5], shape=(1, 5))
        encoder = UniRNNEncoder(UNITS, VOCABS_ZIE, EMBEDDING_SIZE, unit_name='gru')
        output, state = encoder(x)
        self.assertAllEqual([1, 5, UNITS], output.shape)
        self.assertAllEqual([1, UNITS], state.shape)

    def testUniLSTMEncoder(self):
        x = tf.constant([1, 2, 3, 4, 5], shape=(1, 5))
        encoder = UniRNNEncoder(UNITS, VOCABS_ZIE, EMBEDDING_SIZE, unit_name='lstm')
        output, (state_h, state_c) = encoder(x)
        self.assertAllEqual([1, 5, UNITS], output.shape)
        self.assertAllEqual([1, UNITS], state_h.shape)
        self.assertAllEqual([1, UNITS], state_c.shape)

    def testBiGRUEncoder(self):
        x = tf.constant([1, 2, 3, 4, 5], shape=(1, 5))
        # gru in `concat` mode
        encoder = BiRNNEncoder(UNITS, VOCABS_ZIE, EMBEDDING_SIZE, unit_name='gru', merge_mode='concat')
        output, state = encoder(x)
        self.assertAllEqual([1, 5, 2 * UNITS], output.shape)
        self.assertAllEqual([1, 2 * UNITS], state.shape)
        # gru in ['sum', 'ave', 'mul'] mode
        for mode in ['sum', 'ave', 'mul']:
            encoder = BiRNNEncoder(UNITS, VOCABS_ZIE, EMBEDDING_SIZE, unit_name='gru', merge_mode=mode)
            output, state = encoder(x)
            self.assertAllEqual([1, 5, UNITS], output.shape)
            self.assertAllEqual([1, UNITS], state.shape)

    def testBiLSTMEncoder(self):
        x = tf.constant([1, 2, 3, 4, 5], shape=(1, 5))
        # lstm in `concat` mode
        encoder = BiRNNEncoder(UNITS, VOCABS_ZIE, EMBEDDING_SIZE, unit_name='lstm', merge_mode='concat')
        output, (state_h, state_c) = encoder(x)
        self.assertAllEqual([1, 5, 2 * UNITS], output.shape)
        self.assertAllEqual([1, 2 * UNITS], state_c.shape)
        self.assertAllEqual([1, 2 * UNITS], state_h.shape)

        for mode in ['sum', 'ave', 'mul']:
            encoder = BiRNNEncoder(UNITS, VOCABS_ZIE, EMBEDDING_SIZE, unit_name='lstm', merge_mode=mode)
            output, (state_c, state_h) = encoder(x)
            self.assertAllEqual([1, 5, UNITS], output.shape)
            self.assertAllEqual([1, UNITS], state_h.shape)
            self.assertAllEqual([1, UNITS], state_c.shape)


if __name__ == "__main__":
    tf.test.main()
