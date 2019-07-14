import tensorflow as tf
import numpy as np
#mutation_module = tf.load_op_library('/home/jorge/git/custom-op/tensorflow_mutation/python/ops/_mutation_ops.so')


def function_map(xInput, mutationRate):
    return tf.map_fn(lambda x: tf.cond(x < mutationRate, lambda: 1.0, lambda: 0.0), xInput, dtype=tf.float32)

#
# def mutation(tensor,mutationRate,mutationTax):
#     return mutation_module.mutation(tensor);

def mutation(tensor, mutationRate, mutationTax):
    # depois fazer matrix mascara (a.k.a recomendacao do gabriel)
    with tf.name_scope('Mutation'):
        shapeSize = tf.shape(tensor)
        random_array_binary = tf.random_uniform(dtype=tf.float32, minval=0, maxval=100, shape=shapeSize)

        # random_array_binary =  tf.map_fn(lambda x: function_map(x,mutationRate), random_array_binary, dtype=tf.float32)

        # random_array_binary = tf.to_int32(random_array_binary > mutationRate)

        comparison = tf.math.greater(random_array_binary, mutationRate * 100)

        random_array_values = tf.random_uniform(dtype=tf.float32, minval=(1 - mutationTax), maxval=( 1 + mutationTax), shape=shapeSize)

        #random_array_values = tf.multiply(random_array_values, (2 * mutationTax)) + (1 - mutationTax)

        # random_mutation = tf.multiply(random_array_binary, random_array_values)
        # mutated = tf.multiply(tensor, random_mutation)
        #
        # mutated = tf.where(comparison, tensor, mutated)
        mutated = tf.multiply(tensor, random_array_values)

        mutated = tf.where(comparison, tensor, mutated)

        #
        # random_array_values = tf.random_uniform(dtype=tf.float32, shape=shapeSize)
        #
        # random_array_values = tf.multiply(random_array_values, (2 * mutationTax)) + (1 - mutationTax)
        #
        # random_mutation = tf.multiply(random_array_binary, random_array_values)
        #
        # mutated = tf.multiply(tensor, random_mutation)

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
        return mutated;
