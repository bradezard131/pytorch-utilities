import torch


def exponential_moving_average(current_loss, previous_average, alpha = 0.6):
    return alpha * current_loss + (1 - alpha) * previous_average


def harmonic_mean(input):
    return input.pow(-1).mean().pow(-1)
