from numpy import zeros
from numpy import ones
from numpy.random import randn
from keras.models import Sequential, load_model
from keras.layers import Dense
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

# Use the Agg backend
matplotlib.use('Agg')
# Define a specific size for all plots to be the same
plt.rcParams['figure.figsize'] = [10, 5]


def define_discriminator(n_inputs):
    """
    Creates a discriminator model using Keras.
    The discriminator's role is to evaluate the data samples and determine
    whether they are reals (from the training dataset) or fake (produced by the
    generator). It acts as a binary classifier, assigning a probability that a
    given input is real or fake. This implementation consists of an input layer
    whose number of neurons is n_input, a first hidden layer with 20 neurons
    and relu activation function, a second hidden layer with 50 neurons and
    relu activation function, finally an output layer of 1 neuron with sigmoid
    activation function. The loss function is binary crossentropy and the
    optimizer is Adamax.

    n_inputs: number of inputs for the discriminator.
    """
    model = Sequential()
    model.add(Dense(20, activation='relu', kernel_initializer='he_uniform',
                    input_dim=n_inputs))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='Adamax',
                  metrics=['accuracy'])
    return model


def define_generator(latent_dim, n_outputs):
    """
    Creates a generator model using Keras.
    The generator's role is to create new data samples that mimic the
    characteristics of the training data. It starts with random noise
    and uses a neural network to transform this noise into data that
    resembles the real data. The goal of the generator is to produce
    data that the discriminator cannot distinguish from the actual data.
    This implementation consists of an input layer whose number of neurons
    is n_input, a first hidden layer with 256 neurons and relu activation
    function and an output layer of n_output neurons with LeakyReLU activation
    function.

    latent_dim: dimension of a latent vector.
    n_outputs: number of outputs for the generator.
    """
    model = Sequential()
    model.add(Dense(256, activation='relu',  kernel_initializer='he_uniform',
                    input_dim=latent_dim))
    # model.add(Dense(512, activation='relu',  kernel_initializer='he_uniform',
    #                 input_dim=latent_dim))
    model.add(Dense(n_outputs, activation='LeakyReLU'))
    return model


def initialize_models(latent_dim, gen_size, verbose=False):
    """
    Initializes the generator, the discriminator and the GAN model. The
    previously trained models are opened. In the case that they do not exist
    (first time running GAN), the models are created with the corresponding
    functions.

    latent_dim: dimension of every latent vector.
    gen_size: size of the gen (number of chromosmes).
    verbose: is output desired or not.

    generator: generator model.
    discriminator: discriminator model.
    gan: gan model.
    """
    try:
        generator = load_model("./Models/generator_tf")
        if verbose:
            print("Loaded Generator from disk")
    except OSError:
        generator = define_generator(latent_dim, n_outputs=gen_size)

    try:
        discriminator = load_model("./Models/discriminator_tf")
        if verbose:
            print("Loaded Discriminator from disk")
    except OSError:
        discriminator = define_discriminator(n_inputs=gen_size)

    try:
        gan = load_model("./Models/gan_model_tf")
        if verbose:
            print("Loaded GAN_model from disk")
    except OSError:
        gan = define_gan(generator, discriminator)
    return [generator, discriminator, gan]


def define_gan(generator, discriminator):
    """
    Assembles the generator and the discriminator together in a single
    architecture. Defines the discriminator as not trainable in order to train
    separat6ely the discriminator and the generator. The loss function is
    binary crossentropy and the optimizer is Adamax.

    generator: arch of the generator previously defined by define_generator().
    discriminator: architecture of the discriminator previously defined by
    define_discriminator().

    model: combined architecture of generator plus discriminator.
    """
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # Connect them
    model = Sequential()
    # Add generator
    model.add(generator)
    # Add the discriminator
    model.add(discriminator)

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='Adamax')
    return model


def generate_real_samples(n, Data):
    """
    Chooses random values from the original dataset.

    Real_data: subset of the original data.
    n: number of samples to choose.
    """
    random_indices = random.sample(range(len(Data)), n)
    Real_data = Data[random_indices]
    y = ones((n, 1))
    return Real_data, y


def generate_latent_points(latent_dim, n):
    """
    Creates a batch of random points in a latent space. It first generates
    a flat array of random numbers, where the total number of values is the
    product of the latent space dimension and the number of points desired. It
    then reshapes this array into a matrix where each row represents a latent
    vector with the specified dimensionality.

    latent_dim: dimension of every latent vector.
    n: number of vector to create.

    x_input: set of random latent vectors.
    """
    # Generate points in the latent space
    x_input = randn(latent_dim * n)

    # Reshape into a batch of inputs for the network
    x_input = x_input.reshape(n, latent_dim)
    return x_input


def generate_fake_samples(generator, latent_dim, n):
    """
    Generates fake samples predicted by the generator.

    generator: generator model.
    latent_dim: dimension of latent vector.
    n: number of vectors to generate.

    X: generated data.
    y: zeros (tags).
    """
    # Generate points in latent space
    x_input = generate_latent_points(latent_dim, n)
    # Predict outputs
    X = generator.predict(x_input, verbose=0)
    # Create class labels
    y = zeros((n, 1))
    return X, y


def plot_dist(real_data, fake_data, epoch, param):
    """
    Generates and saves a histogram comparing real and synthetic data
    distributions.

    Parameters:
    real_data : array-like
        The real data sample.
    fake_data : array-like
        The synthetic data sample.
    epoch : int
        Current epoch number.
    param : int
        The index of the evaluated column in the dataset.

    Returns:
    None
    """
    # Ensure the epoch is at least 1
    if epoch == 0:
        epoch = 1

    # Set plot title with epoch number
    plt.title(f"Epoch: {epoch}")

    # Plot histograms for real and fake data
    sns.histplot(x=fake_data, kde=True, color="blue", stat="density",
                 label="Fake")
    sns.histplot(x=real_data, kde=True, color="red", stat="density",
                 label="Real")

    # Determine x-axis limits
    x_min = min(min(real_data), min(fake_data)) - 2
    x_max = max(max(real_data), max(fake_data)) + 2
    plt.xlim([x_min, x_max])

    # Add legend and labels
    plt.legend()
    plt.xlabel(f"Parameter {param + 1}")

    # Save the figure as a PDF
    output_filename = f"./Histograms/Hist_Param{param + 1}_Epoch{epoch}.pdf"
    plt.savefig(output_filename)

    # Close the plot to free memory
    plt.close()


def plot_history(d_hist, g_hist):
    """
    Creates a figure with the loss history of the network.

    d_hist: shistoric loss of the discriminator.
    g_hist: historic loss of the generator.
    """
    plt.plot(g_hist[1:], label='Generator', color='blue')
    plt.plot(d_hist[1:], label='Discriminator', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"./Loss/Loss_Epoch{len(g_hist)}.pdf")
    plt.close()


def summarize_performance(epoch, generator, latent_dim, n, Data):
    """
    Summarizes the performance of the generator compering it with the real data.

    epoch: current epoch.
    generator: generator model.
    latent_dim: dimension of latent vector.
    n: number of individuals.
    Data: original data.
    """
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, int(n*20))

    # Plots frequency distributions
    for col in range(len(Data[0])):
        plot_dist(Data[:, col], x_fake[:, col], epoch, col)

    return x_fake


def train(g_model, d_model, gan_model, latent_dim, Data, n_epochs, n_eval=250,
          verbose=False):
    """
    Trains the GAN.

    g_model: generator model.
    d_model: discriminator model.
    gan_model: complete GAN model.
    latent_dim: dimension of a latent vector.
    Data: Original data.
    n_epochs: number of epochs to train in total.
    n_eval: size of intervals before every evaluation.

    x_fake_last: set of synthetic data obtained on the last evaluation.
    """

    d_history_loss = []
    g_history_loss = []
    n_batch = len(Data)  # Number of individuals
    # manually enumerate epochs
    for i in range(n_epochs+1):
        # Prepare real samples
        n_real_samples = int(n_batch*0.5)
        x_real, y_real = generate_real_samples(n_real_samples, Data)
        # Prepare fake examples
        n_f_samples = int(n_batch*0.7)
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim,
                                               n_f_samples)

        # Update discriminator
        d_loss_real, _ = d_model.train_on_batch(x_real, y_real)
        d_loss_fake, _ = d_model.train_on_batch(x_fake, y_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Prepare points in latent space as input for the generator
        x_gan = generate_latent_points(latent_dim, n_batch)
        # Create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # Update the generator via the discriminator's error
        g_loss_fake = gan_model.train_on_batch(x_gan, y_gan)

        # Append to historic loss values
        d_history_loss.append(d_loss)
        g_history_loss.append(g_loss_fake)

        # Evaluate the model every n_eval epochs
        if ((i) % n_eval) == 0 and verbose:
            x_fake_last = summarize_performance(i, g_model, latent_dim,
                                                n_batch, Data)
            # Plots loss history
            plot_history(d_history_loss, g_history_loss)
        # Creates the returning fakes samples for the last iteration
        if i+1 == n_epochs:
            x_fake_last, y_fake = generate_fake_samples(g_model, latent_dim,
                                                        n_batch)
    return x_fake_last


def save_models(generator, discriminator, gan_model):
    """
    Saves the models in memory.

    generator: model of generator.
    discriminator: model of discriminator.
    """
    # Saving the model in tensorflow format
    generator.save('./Models/generator_tf', save_format='tf')

    # Saving the model in tensorflow format
    discriminator.save('./Models/discriminator_tf', save_format='tf')

    # Saving the model in tensorflow format
    gan_model.save('./Models/gan_model_tf', save_format='tf')

    print("Saved models to disk")


def generate_individuals(Data):
    Data = np.array(Data)  # Apropiate format to original dataset
    gen_size = len(Data[0])  # Size of a single individual
    latent_dim = 128  # Size of the latent space

    # Initializes models
    generator, discriminator, gan = initialize_models(latent_dim, gen_size)

    # Train model
    Data_fakeX = train(generator, discriminator, gan,
                       latent_dim, Data, n_epochs=20000, n_eval=400,
                       verbose=True)

    save_models(generator, discriminator, gan)

    return Data_fakeX


if __name__ == "__main__":
    # Load the dataset from the Excel file
    input_filename = 'random_dataset.xlsx'
    df = pd.read_excel(input_filename)

    # Convert the DataFrame to a NumPy array
    np_array = df.to_numpy()

    # Flatten the NumPy array to 1D
    # flat_array = np_array.flatten()
    generate_individuals(np_array)
