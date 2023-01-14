
# JaxCV

JaxCV is like OpenCV-on-Steriods and based on Jax and amazing XLA compiler.
Its long term goal is to build a de-facto computer vision library for research and production at scale
#### Features:
 
    1. Algorithms are composable as direct-asyclic-graphs. Easy to implement complex pipelines.
    2. Based on high efficient classical algorithms, majority of the algorithms runs in few millis
    3. Most of the algorithms are differentiable
    4. Supports vertical and horizontal scaling of algorithms (BxHxWxC), on an entire batch of images
    5. Composed transorfmations friendly as layers with Flax and Haiku
    5. Composed transorfmations can be convertible to TFLITE for easy production, 
        a strong contender to MediaPipe
    6. Algorithms can be exported as Tensorflow SavedModel
    7. Meaningful APIs and errors

----

```python
import jax
import jaxcv as jcv

composer = jcv.Compose(
  jcv.augment.RandomCrop(256, 256),
  jcv.augment.HorizontalFlip(),
  jcv.augment.RandomRotate(min=0, max=130),
)

image_batch =  p = jax.device_put(jax.numpy.ones(100, 640, 480, 3), device=jax.devices('gpu')[0])

rng = jax.random.PRNGKey(27)

transformed_image = composer(rng, image_batch)

composer.export.to_tflite('augment_composer.tflite')
                
----

```

This repo is in its first commit. 
I'll regularly update with many computer vision algorithms but takes significant efforts.
Stay Tuned.

All the algorithms have been testing on RTX 2070 GPU with intel-i7 10th Gen.