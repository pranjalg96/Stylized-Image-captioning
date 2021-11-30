import pickle
import matplotlib.pyplot as plt

disc_losses = None
gen_losses = None
enc_losses = None
disc_accuracies = None

with open('Results/supervised/disc_losses.pkl', 'rb') as file:
    disc_losses = pickle.load(file)

with open('Results/supervised/gen_losses.pkl', 'rb') as file:
    gen_losses = pickle.load(file)

with open('Results/supervised/enc_losses.pkl', 'rb') as file:
    enc_losses = pickle.load(file)

with open('Results/supervised/disc_accuracies.pkl', 'rb') as file:
    disc_accuracies = pickle.load(file)

plt.plot(disc_losses)
plt.xlabel('Epochs')
plt.ylabel('Disc loss')
plt.show()

plt.plot(gen_losses)
plt.xlabel('Epochs')
plt.ylabel('Gen. loss')
plt.show()

plt.plot(enc_losses)
plt.xlabel('Epochs')
plt.ylabel('Enc. loss')
plt.show()

plt.plot(disc_accuracies)
plt.xlabel('Epochs')
plt.ylabel('Disc. Accuracies')
plt.show()

