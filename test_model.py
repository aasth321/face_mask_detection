import tensorflow as tf

mask_model = tf.keras.models.load_model("mask_detector.h5")

print("\n=== MODEL SUMMARY DETAILS ===")
mask_model.summary()

print("\n=== FINAL OUTPUT LAYER SHAPE ===")
print(mask_model.layers[-1].output_shape)