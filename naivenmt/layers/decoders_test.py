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

from naivenmt.layers.decoders import BiRNNDecoder, UniRNNDecoder

BATCH_SIZE = 2
SRC_SEQ_LEN = 3
TGT_SEQ_LEN = 1
UNITS = 16
TGT_VOCAB_SIZE = 50
EMBEDDING_SIZE = 32


class DecodersTest(tf.test.TestCase):

    def testUniGRUDecoder(self):
        x = tf.constant([1, 2], shape=(BATCH_SIZE, TGT_SEQ_LEN))
        initial_state = tf.zeros((BATCH_SIZE, UNITS))
        enc_output = tf.random.uniform((BATCH_SIZE, SRC_SEQ_LEN,  UNITS))
        for attention in ['bahdanau', 'luong']:
            decoder = UniRNNDecoder(UNITS, TGT_VOCAB_SIZE, EMBEDDING_SIZE, unit_name='gru', attention=attention)
            output, state, attn_weights = decoder(x, state=initial_state, enc_output=enc_output)

            self.assertAllEqual([BATCH_SIZE, TGT_VOCAB_SIZE], output.shape)
            self.assertAllEqual([BATCH_SIZE, UNITS], state.shape)
            self.assertAllEqual([BATCH_SIZE, SRC_SEQ_LEN], attn_weights.shape)

    def testUniLSTMDecoder(self):
        x = tf.constant([1, 2], shape=(BATCH_SIZE, TGT_SEQ_LEN))
        initial_state = tf.zeros((BATCH_SIZE, UNITS))
        enc_output = tf.random.uniform((BATCH_SIZE, SRC_SEQ_LEN,  UNITS))
        for attention in ['bahdanau', 'luong']:
            decoder = UniRNNDecoder(UNITS, TGT_VOCAB_SIZE, EMBEDDING_SIZE, unit_name='lstm', attention=attention)
            output, (state_h, state_c), attn_weights = decoder(
                x, state=(initial_state, initial_state), enc_output=enc_output)

            self.assertAllEqual([BATCH_SIZE, TGT_VOCAB_SIZE], output.shape)
            self.assertAllEqual([BATCH_SIZE, UNITS], state_h.shape)
            self.assertAllEqual([BATCH_SIZE, UNITS], state_c.shape)
            self.assertAllEqual([BATCH_SIZE, SRC_SEQ_LEN], attn_weights.shape)

    def testBiGRUDecoder(self):
        x = tf.constant([1, 2], shape=(BATCH_SIZE, TGT_SEQ_LEN))
        initial_state = tf.zeros((BATCH_SIZE, UNITS))
        enc_output = tf.random.uniform((BATCH_SIZE, SRC_SEQ_LEN,  UNITS))
        for attention in ['bahdanau', 'luong']:
            decoder = BiRNNDecoder(UNITS, TGT_VOCAB_SIZE, EMBEDDING_SIZE, unit_name='gru',
                                   attention=attention, merge_mode='concat')
            output, (state_h, state_c), attn_weights = decoder(
                x, state=(initial_state, initial_state),
                enc_output=enc_output)

            self.assertAllEqual([BATCH_SIZE, TGT_VOCAB_SIZE], output.shape)
            self.assertAllEqual([BATCH_SIZE, UNITS], state_h.shape)
            self.assertAllEqual([BATCH_SIZE, UNITS], state_c.shape)
            self.assertAllEqual([BATCH_SIZE, SRC_SEQ_LEN], attn_weights.shape)

            for mode in ['ave', 'sum', 'mul']:
                decoder = BiRNNDecoder(UNITS, TGT_VOCAB_SIZE, EMBEDDING_SIZE, unit_name='gru',
                                       attention=attention, merge_mode=mode)
                output, (state_h, state_c), attn_weights = decoder(
                    x, state=(initial_state, initial_state),
                    enc_output=enc_output)

                self.assertAllEqual([BATCH_SIZE, TGT_VOCAB_SIZE], output.shape)
                self.assertAllEqual([BATCH_SIZE, UNITS], state_h.shape)
                self.assertAllEqual([BATCH_SIZE, UNITS], state_c.shape)
                self.assertAllEqual([BATCH_SIZE, SRC_SEQ_LEN], attn_weights.shape)

    def testBiLSTMDecoder(self):
        x = tf.constant([1, 2], shape=(BATCH_SIZE, TGT_SEQ_LEN))
        initial_state = tf.zeros((BATCH_SIZE, UNITS))
        enc_output = tf.random.uniform((BATCH_SIZE, SRC_SEQ_LEN,  UNITS))
        for attention in ['bahdanau', 'luong']:
            decoder = BiRNNDecoder(UNITS, TGT_VOCAB_SIZE, EMBEDDING_SIZE, unit_name='lstm',
                                   attention=attention, merge_mode='concat')
            output, (state_fw_h, state_fw_c, state_bw_h, state_bw_c), attn_weights = decoder(
                x, state=(initial_state, initial_state, initial_state, initial_state), enc_output=enc_output)

            self.assertAllEqual([BATCH_SIZE, TGT_VOCAB_SIZE], output.shape)
            self.assertAllEqual([BATCH_SIZE, UNITS], state_fw_h.shape)
            self.assertAllEqual([BATCH_SIZE, UNITS], state_fw_c.shape)
            self.assertAllEqual([BATCH_SIZE, UNITS], state_bw_h.shape)
            self.assertAllEqual([BATCH_SIZE, UNITS], state_bw_c.shape)
            self.assertAllEqual([BATCH_SIZE, SRC_SEQ_LEN], attn_weights.shape)

            for mode in ['ave', 'sum', 'mul']:
                decoder = BiRNNDecoder(UNITS, TGT_VOCAB_SIZE, EMBEDDING_SIZE, unit_name='lstm',
                                       attention=attention, merge_mode=mode)
                output, (state_fw_h, state_fw_c, state_bw_h, state_bw_c), attn_weights = decoder(
                    x, state=(initial_state, initial_state, initial_state, initial_state), enc_output=enc_output)

                self.assertAllEqual([BATCH_SIZE, TGT_VOCAB_SIZE], output.shape)
                self.assertAllEqual([BATCH_SIZE, UNITS], state_fw_h.shape)
                self.assertAllEqual([BATCH_SIZE, UNITS], state_fw_c.shape)
                self.assertAllEqual([BATCH_SIZE, UNITS], state_bw_h.shape)
                self.assertAllEqual([BATCH_SIZE, UNITS], state_bw_c.shape)
                self.assertAllEqual([BATCH_SIZE, SRC_SEQ_LEN], attn_weights.shape)


if __name__ == "__main__":
    tf.test.main()
