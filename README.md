# willseyetal2022
Contains the code defining the temporally-convolved feedforward neural network used in Willsey et al. 2022 (Nat Comms).

The neural network is initialized as follows:

```
# NN variables
input_size = 96
hidden_size = 256
ConvSizeOut = 16 #16
ConvSize = 3
num_states = 2


# Define the network
model = NNDecoders.FC4L256Np05_CNN1L16N_SBP(input_size, hidden_size, ConvSize, ConvSizeOut, num_states).to(device)
```

Notes:
- Despite less than 96 channels being used, the neural network is still defined with 96 inputs. Unused channels are multiplied by zero before being sent to the network. Due to this approach, the 'BadChannels' parameter of the forward pass of FC4L256Np05_CNN1L16N_SBP is not used.
