"""
Quick test script for Colab - Test your trained Cigdem TTS model
Copy this into a new Colab cell to test your model
"""

# Install required package if not already installed
!pip install -q phonemizer

# Test your model with Turkish text
import sys
sys.path.append('/content/TTS_Cigdem')

# Run inference
!python inference.py \
    --checkpoint "Models/Cigdem_TTS/epoch_2nd_00070.pth" \
    --text "Merhaba, ben Cigdem. Nasılsınız?" \
    --output "test_output.wav"

# Play the generated audio
from IPython.display import Audio, display
display(Audio("test_output.wav", autoplay=True))

print("\n✅ Audio generated! You should hear Cigdem's voice.")
print("To test with your own text, run:")
print('!python inference.py -t "Your Turkish text here" -o output.wav')
