import argparse
import sys
import os

import tensorflow as tf

from tensorflow.python.framework import graph_util

#---------------------------------------------------------------------------
# Parse the commandline
#---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Export a tensorflow model')
parser.add_argument('--metagraph-file', default='final.ckpt.meta',
                    help='name of the metagraph file')
parser.add_argument('--checkpoint-file', default='final.ckpt',
                    help='name of the checkpoint file')
parser.add_argument('--output-file', default='model.pb',
                    help='name of the output file')
parser.add_argument('--output-tensors', nargs='+',
                    required=True,
                    help='names of the output tensors')
args = parser.parse_args()

print('[i] Matagraph file:  ', args.metagraph_file)
print('[i] Checkpoint file: ', args.checkpoint_file)
print('[i] Output file:     ', args.output_file)
print('[i] Output tensors:  ', args.output_tensors)

for f in [args.checkpoint_file+'.index', args.metagraph_file]:
    if not os.path.exists(f):
        print('[!] Cannot find file:', f)
        sys.exit(1)

#-------------------------------------------------------------------------------
# Export the graph
#-------------------------------------------------------------------------------
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(args.metagraph_file)
    saver.restore(sess, args.checkpoint_file)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, args.output_tensors)

    with open(args.output_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())
