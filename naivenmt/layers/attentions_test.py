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

from naivenmt.layers.attentions import BahdanauAttention, LuongAttention

UNITS = 32
BATCH_SIZE = 2
SRC_SEQ_LEN = 10


class AttentionsTest(tf.test.TestCase):

    def testBahdanauAttention(self):
        attention = BahdanauAttention(UNITS)
        dec_state = tf.random.uniform((BATCH_SIZE, UNITS))
        enc_output = tf.random.uniform((BATCH_SIZE, SRC_SEQ_LEN, UNITS))

        context, attn_weights = attention(dec_state, enc_output)

        self.assertAllEqual([BATCH_SIZE, UNITS], context.shape)
        self.assertAllEqual([BATCH_SIZE, SRC_SEQ_LEN], attn_weights.shape)

    def testLuongAttention(self):
        attention = LuongAttention(UNITS)
        dec_output = tf.random.uniform((BATCH_SIZE, 1, UNITS))
        enc_output = tf.random.uniform((BATCH_SIZE, SRC_SEQ_LEN, UNITS))

        context, attn_weights = attention(dec_output, enc_output)

        self.assertAllEqual([BATCH_SIZE, UNITS], context.shape)
        self.assertAllEqual([BATCH_SIZE, SRC_SEQ_LEN], attn_weights.shape)


if __name__ == "__main__":
    tf.test.main()
