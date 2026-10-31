from machine import Pin
from inference import predict

# LEDs: each LED must have its own 220-330 ohm current-limiting resistor.
led1 = Pin(10, Pin.OUT)
led2 = Pin(11, Pin.OUT)
result_led = Pin(12, Pin.OUT)

print("TinyML XOR demo running on Raspberry Pi Pico 2.")
print("Enter 00, 01, 10, or 11 in the Thonny Shell. Enter q to stop.")

while True:
    try:
        user_input = input("Input (00/01/10/11, q to quit): ").strip()

        if user_input.lower() == "q":
            led1.off()
            led2.off()
            result_led.off()
            print("Stopped.")
            break

        if user_input not in ("00", "01", "10", "11"):
            print("Invalid input. Enter 00, 01, 10, or 11.")
            continue

        x1 = int(user_input[0])
        x2 = int(user_input[1])

        prediction, probability = predict([x1, x2])

        # First two LEDs show the entered input bits.
        led1.value(x1)
        led2.value(x2)

        # Third LED shows the neural network prediction.
        result_led.value(prediction)

        print("Input:", x1, x2)
        print("Prediction:", prediction)
        print("Probability:", probability)

    except KeyboardInterrupt:
        led1.off()
        led2.off()
        result_led.off()
        print("\nStopped.")
        break
