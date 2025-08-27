import pandas as pd

def create_test_dataset():
    """
    Defines the full test dataset and saves it to a CSV file.
    """
    test_data = [
        # --- Questions for L5.pdf (Separation of Substances) ---
        # Category 1: Basic Retrieval
        {
            "question": "What is the process of condensation?",
            "ground_truth": "Condensation is the process of conversion of water vapour into its liquid form."
        },
        {
            "question": "For what purpose is the method of 'winnowing' used?",
            "ground_truth": "Winnowing is used to separate heavier and lighter components of a mixture by wind or by blowing air."
        },
        # Category 2: Plausible but Incorrect
        {
            "question": "What is the chemical formula for salt used in the evaporation experiment?",
            "ground_truth": "The provided context does not contain information about the chemical formula for salt."
        },
        # Category 3: Zero-Context
        {
            "question": "What is the primary ingredient in sourdough bread?",
            "ground_truth": "I cannot answer that question based on the provided context."
        },

        # --- Questions for L6.pdf (Changes Around Us) ---
        # Category 1: Basic Retrieval
        {
            "question": "What happens when you heat a ring on the 'metal ring' apparatus?",
            "ground_truth": "When the ring is heated, it expands and becomes slightly larger in size."
        },
        {
            "question": "Can the change from rolling out a roti from a ball of dough be reversed?",
            "ground_truth": "Yes, the change can be reversed by changing the rolled roti back into a ball of dough."
        },
        # Category 2: Plausible but Incorrect
        {
            "question": "What is the melting point of the wax candle mentioned in the chapter?",
            "ground_truth": "I don't know."
        },
        # Category 3: Zero-Context
        {
            "question": "Who is the current CEO of Microsoft?",
            "ground_truth": "I cannot answer that question based on the provided context."
        },

        # --- Questions for L7.pdf (Getting to Know Plants) ---
        # Category 1: Basic Retrieval
        {
            "question": "What are the parts of a stamen in a flower?",
            "ground_truth": "The parts of a stamen are the anther and the filament."
        },
        {
            "question": "What is transpiration?",
            "ground_truth": "Transpiration is the process where water comes out of plant leaves in the form of vapour."
        },
        # Category 2: Plausible but Incorrect
        {
            "question": "What is the role of chlorophyll in photosynthesis according to the text?",
            "ground_truth": "I apologize, but I cannot answer that question based on the provided context."
        },
        # Category 3: Zero-Context
        {
            "question": "What is the distance from the Earth to the Moon?",
            "ground_truth": "I don't know."
        },

        # --- Questions for L8.pdf (Body Movements) ---
        # Category 1: Basic Retrieval
        {
            "question": "What kind of joint allows the head to bend forward and backward, and turn to the right or left?",
            "ground_truth": "The pivotal joint allows the head to bend forward and backward and turn to the right or left."
        },
        {
            "question": "How does a snail move?",
            "ground_truth": "A snail moves with the help of a muscular foot, which produces a slimy substance called mucus to reduce friction."
        },
        # Category 2: Plausible but Incorrect
        {
            "question": "How many bones are in the human hand as detailed in the chapter?",
            "ground_truth": "The provided context does not contain information about the specific number of bones in the human hand."
        },
        # Category 3: Zero-Context
        {
            "question": "What is the main function of a car's radiator?",
            "ground_truth": "I cannot answer that question based on the provided context."
        },

        # --- Questions for L9.pdf (The Living Organisms and Their Surroundings) ---
        # Category 1: Basic Retrieval
        {
            "question": "What are abiotic factors in a habitat?",
            "ground_truth": "The non-living things such as rocks, soil, air and water in a habitat are its abiotic factors."
        },
        {
            "question": "How are the bodies of fish streamlined?",
            "ground_truth": "The head and tail of the fish are smaller than the middle portion of the body, creating a streamlined shape."
        },
        # Category 2: Plausible but Incorrect
        {
            "question": "What is the average lifespan of the desert rats mentioned in the text?",
            "ground_truth": "I don't know."
        },
        # Category 3: Zero-Context
        {
            "question": "What is the recipe for pasta carbonara?",
            "ground_truth": "I cannot answer that question based on the provided context."
        },

        # --- Questions for L10.pdf (Motion and Measurement of Distances) ---
        # Category 1: Basic Retrieval
        {
            "question": "What is a 'cubit' as a unit of measurement?",
            "ground_truth": "A cubit is the length from the elbow to the fingertips."
        },
        {
            "question": "What is rectilinear motion?",
            "ground_truth": "Motion in a straight line is called rectilinear motion."
        },
        # Category 2: Plausible but Incorrect
        {
            "question": "What was the year the metric system was created by the French, according to the document?",
            "ground_truth": "The provided context does not contain information about the specific year the metric system was created."
        },
        # Category 3: Zero-Context
        {
            "question": "What is the capital of Australia?",
            "ground_truth": "I apologize, but I cannot answer that question based on the provided context."
        },

        # --- Questions for L11.pdf (Light, Shadows and Reflections) ---
        # Category 1: Basic Retrieval
        {
            "question": "What is a luminous object?",
            "ground_truth": "Objects that give out or emit light of their own are called luminous objects."
        },
        {
            "question": "How is the image formed by a pinhole camera oriented?",
            "ground_truth": "The image formed by a pinhole camera is upside down."
        },
        # Category 2: Plausible but Incorrect
        {
            "question": "What is the speed of light as stated in this chapter?",
            "ground_truth": "I don't know."
        },
        # Category 3: Zero-Context
        {
            "question": "Who invented the telephone?",
            "ground_truth": "I cannot answer that question based on the provided context."
        },

        # --- Questions for L12.pdf (Electricity and Circuits) ---
        # Category 1: Basic Retrieval
        {
            "question": "What are the two terminals of an electric cell?",
            "ground_truth": "An electric cell has two terminals: a positive terminal (+) and a negative terminal (–)."
        },
        {
            "question": "What is the purpose of a switch in an electric circuit?",
            "ground_truth": "A switch is a simple device that either breaks the circuit or completes it, turning the bulb on or off."
        },
        # Category 2: Plausible but Incorrect
        {
            "question": "What is the voltage of the electric cell shown in the diagrams?",
            "ground_truth": "The provided context does not contain information about the specific voltage of the electric cell."
        },
        # Category 3: Zero-Context
        {
            "question": "What is the process of photosynthesis?",
            "ground_truth": "I apologize, but I cannot answer that question based on the provided context."
        },

        # --- Questions for L13.pdf (Fun with Magnets) ---
        # Category 1: Basic Retrieval
        {
            "question": "In which direction does a freely suspended bar magnet always come to rest?",
            "ground_truth": "A freely suspended bar magnet always comes to rest in the north-south direction."
        },
        {
            "question": "What is the property of 'repulsion' between magnets?",
            "ground_truth": "Repulsion is the property where like poles of two magnets repel each other."
        },
        # Category 2: Plausible but Incorrect
        {
            "question": "Who was the shepherd named Magnes who discovered magnets, according to the legend?",
            "ground_truth": "I don't know."
        },
        # Category 3: Zero-Context
        {
            "question": "What are the primary colors in additive color mixing?",
            "ground_truth": "I cannot answer that question based on the provided context."
        }
    ]
    
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(test_data)
    
    # Define the output filename
    output_filename = "test_dataset.csv"
    
    # Save the DataFrame to a CSV file, without the pandas index
    df.to_csv(output_filename, index=False)
    
    print(f"✅ Successfully created '{output_filename}' with {len(df)} questions.")


if __name__ == "__main__":
    create_test_dataset()