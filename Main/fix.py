import csv

# --- Configuration ---
# Your original CSV file
input_filename = 'dataset.csv'
# The new file that will be created
output_filename = 'dataset2.csv'
# The index of the column you want to remove (0 is the first, 1 is the second, 2 is the third)
column_index_to_remove = 2

# --- Processing ---
try:
    with open(input_filename, 'r', newline='', encoding='utf-8') as infile, \
         open(output_filename, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            # Make sure the row has enough columns before trying to remove one
            if len(row) > column_index_to_remove:
                # Remove the element at the specified index
                del row[column_index_to_remove]
                # Write the modified row to the new file
                writer.writerow(row)

    print(f"✅ Successfully removed column {column_index_to_remove + 1} and saved the result to '{output_filename}'.")

except FileNotFoundError:
    print(f"❌ Error: The file '{input_filename}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")