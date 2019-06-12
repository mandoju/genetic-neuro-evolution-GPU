import tensorflow as tf
import numpy as np


def function_map(xInput, mutationRate):
    return tf.map_fn(lambda x: tf.cond(x < mutationRate, lambda: 1.0, lambda: 0.0), xInput, dtype=tf.float32)


def mutation(tensor, mutationRate, mutationTax):
    # depois fazer matrix mascara (a.k.a recomendacao do gabriel)
    with tf.name_scope('Mutation'):
        shapeSize = tf.shape(tensor)
        random_array_binary = tf.random_uniform(dtype=tf.float32, minval=0, maxval=100, shape=shapeSize)

        # random_array_binary =  tf.map_fn(lambda x: function_map(x,mutationRate), random_array_binary, dtype=tf.float32)

        # random_array_binary = tf.to_int32(random_array_binary > mutationRate)

        comparison = tf.math.greater(random_array_binary, mutationRate * 100)

        random_array_binary = tf.where(comparison, tf.zeros_like(random_array_binary),
                                       tf.ones_like(random_array_binary))

        random_array_values = tf.random_uniform(dtype=tf.float32, shape=shapeSize)

        random_array_values = tf.multiply(random_array_values, (2 * mutationTax)) + (1 - mutationTax)

        random_mutation = tf.multiply(random_array_binary, random_array_values)

        mutated = tf.multiply(tensor, random_mutation)
        # comparison = tf.math.equal( random_mutation, tf.constant( 0.0 ) )
        # mutated = tf.where(comparison, tensor, random_mutation)
        return mutated


def mutation_unbiased(tensor, mutationRate, mutationTax):
    # depois fazer matrix mascara (a.k.a recomendacao do gabriel)
    with tf.name_scope('Mutation_Unbiased'):
        shapeSize = tf.shape(tensor)
        random_array_binary = tf.random_uniform(dtype=tf.float32, minval=0, maxval=100, shape=shapeSize)

        # random_array_binary =  tf.map_fn(lambda x: function_map(x,mutationRate), random_array_binary, dtype=tf.float32)

        # random_array_binary = tf.to_int32(random_array_binary > mutationRate)

        comparison = tf.math.greater(random_array_binary, mutationRate * 100)

        random_array_binary = tf.where(comparison, tensor,
                                       tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=shapeSize))

        # random_array_values =  tf.random_uniform(dtype=tf.float32, minval=(1-mutationTax), maxval=(1+mutationTax), shape=shapeSize)

        # random_mutation = tf.multiply(random_array_binary,random_array_values)

        # mutated = tf.multiply(tensor,random_mutation)
        # comparison = tf.math.equal( random_mutation, tf.constant( 0.0 ) )
        # mutated = tf.where(comparison, tensor, random_mutation)
        return random_array_binary


def mutation_fixed_number(tensor, mutationRate, mutationTax):
    with tf.name_scope('Mutation'):
        shapeSize = tf.shape(tensor)
        random_array_binary = tf.random_uniform(dtype=tf.float32, minval=0, maxval=100, shape=shapeSize)

        size_mutate_tensor = tf.size(tensor) * mutationRate

        random_array_idexes = tf.random_unifom(dtype=tf.float32)

        random_array_values = tf.random_uniform(dtype=tf.float32, shape=shapeSize)

        random_array_values = tf.multiply(random_array_values, (2 * mutationTax)) + (1 - mutationTax)

        random_mutation = tf.multiply(random_array_binary, random_array_values)

        mutated = tf.multiply(tensor, random_mutation)
        return mutated


def mutation_by_node(tensor, mutationRate):
    # depois fazer matrix mascara (a.k.a recomendacao do gabriel)
    with tf.name_scope('Mutation_node'):
        shapeSize = tf.shape(tensor)
        random_array_binary = tf.random_uniform(dtype=tf.float32, minval=0, maxval=100,
                                                shape=[shapeSize[0], shapeSize[1]])

        # random_array_binary =  tf.map_fn(lambda x: function_map(x,mutationRate), random_array_binary, dtype=tf.float32)

        # random_array_binary = tf.to_int32(random_array_binary > mutationRate)

        comparison = tf.math.greater(random_array_binary, tf.constant(mutationRate * 100))

        random_array_binary = tf.where(comparison, tf.zeros_like(random_array_binary),
                                       tf.ones_like(random_array_binary))

        random_array_values = tf.random_uniform(dtype=tf.float32, minval=-1, maxval=1, shape=shapeSize)

        random_mutation = tf.multiply(random_array_binary, random_array_values)

        mutated = tensor + random_mutation
        # comparison = tf.math.equal( random_mutation, tf.constant( 0.0 ) )
        # mutated = tf.where(comparison, tensor, random_mutation)
        return mutated


def random_choice(a, axis, samples_shape=None):
    """

    :param a: tf.Tensor
    :param axis: int axis to sample along
    :param samples_shape: (optional) shape of samples to produce. if not provided, will sample once.
    :returns: tf.Tensor of shape a.shape[:axis] + samples_shape + a.shape[axis + 1:]
    :rtype:

    Examples:
    >>> a = tf.placeholder(shape=(10, 20, 30), dtype=tf.float32)
    >>> random_choice(a, axis=0)
    <tf.Tensor 'GatherV2:0' shape=(1, 20, 30) dtype=float32>
    >>> random_choice(a, axis=1)
    <tf.Tensor 'GatherV2_1:0' shape=(10, 1, 30) dtype=float32>
    >>> random_choice(a, axis=1, samples_shape=(2, 3))
    <tf.Tensor 'GatherV2_2:0' shape=(10, 2, 3, 30) dtype=float32
    >>> random_choice(a, axis=0, samples_shape=(100,))
    <tf.Tensor 'GatherV2_3:0' shape=(100, 20, 30) dtype=float32>
    """

    if samples_shape is None:
        samples_shape = (1,)
    shape = tuple(a.get_shape().as_list())
    dim = shape[axis]
    choice_indices = tf.random_uniform(samples_shape, minval=0, maxval=dim, dtype=tf.int32)
    samples = tf.gather(a, choice_indices, axis=axis)
    return samples
