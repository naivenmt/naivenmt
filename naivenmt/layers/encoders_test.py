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

    def testUniRNNEncoder(self):
        unit_names = ['gru', 'lstm']
        x = tf.constant([1, 2, 3, 4, 5], shape=(1, 5))
        for unit_name in unit_names:
            encoder = UniRNNEncoder(UNITS, VOCABS_ZIE, EMBEDDING_SIZE, unit_name=unit_name)
            initial_state = encoder.create_initial_state(x)
            output, state = encoder(x, initial_state=initial_state)
            self.assertAllEqual([1, 5, UNITS], output.shape)
            self.assertAllEqual([1, UNITS], state.shape)
            print('======Passed %s unit.\n' % unit_name)

    def testBiRNNEncoder(self):
        encoder = BiRNNEncoder(UNITS, VOCABS_ZIE, EMBEDDING_SIZE, unit_name='gru', merge_mode='concat')
        x = tf.constant([1, 2, 3, 4, 5], shape=(1, 5))
        output, state = encoder(x, initial_state=encoder.create_initial_state(x))
        print('output shape: ', output.shape)
        print('state shape: ', state.shape)
        self.assertAllEqual([1, 5, 2 * UNITS], output.shape)
        self.assertAllEqual([1, UNITS], state.shape)


if __name__ == "__main__":
    tf.test.main()
