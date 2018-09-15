import tensorflow as tf
import os
import time

millis = int(round(time.time() * 1000))
# Idea is that maybe you can set the model id based on the time of creation

# uncomment and set the GPU id if applicable.
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

tf.app.flags.DEFINE_integer('model_version', 1, 'Models version number.')
tf.app.flags.DEFINE_string('work_dir', './tflow', 'Working directory.')
tf.app.flags.DEFINE_integer('model_id', 1, 'Model id name to be loaded.')
tf.app.flags.DEFINE_string('export_model_dir', "./tflow/models", 'Directory where the model exported files should be placed.')

FLAGS = tf.app.flags.FLAGS

# Export model
export_path = os.path.join(
    tf.compat.as_bytes(FLAGS.export_model_dir),
    tf.compat.as_bytes(str(FLAGS.model_version)))

print('Trying to export trained model to:', export_path)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

# Build the signature_def_map.
# Requires access to examples and predictions
regression_signature = tf.saved_model.signature_def_utils.regression_signature_def(
    examples,
    predictions
)

builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            regression_signature,
    },
    main_op=tf.tables_initializer(),
    strip_default_attrs=True)

builder.save()

print('Done exporting!')
