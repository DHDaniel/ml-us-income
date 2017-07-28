
import csv

fhandler = open("adult.data.txt", "r")
reader = csv.reader(fhandler, delimiter=",")

write_handler = open("adult.data.processed.csv", "w")
writer = csv.writer(write_handler, delimiter=",")


# the value map dictionary has the keys of all known categories that have text values (and those values are finite). When the script encounters a value corresponding to one of these classes, it will check to see if the value is already inside the corresponding array. If it is, then it will take the index of that value and add it to the row to be written. If it is not in yet, then it will add the value into the array, and take the index of the value (same as the other case). The only exception is the "income" category, which will already have predetermined indexes for the values.
value_map = {"workclass" : [], "education" : [], "marital-status" : [], "occupation" : [], "relationship" : [], "race" : [], "sex" : [], "native-country" : [], "income" : ["<=50K", ">50K"]}

# reference for categories that each value belongs to
headers = []
# flag to indicate first row (headers)
first_row = True
for row in reader:

    # if header row, we make our reference list
    if first_row:
        for header in row:
            header = header.strip()
            headers.append(header)
        first_row = False
        # skip this row. Do not write headers to processed file.
        continue

    # if not, we prepare a row to be written to the processed file
    row_to_write = []

    for index, value in enumerate(row):

        label = headers[index]
        value = value.strip()
        new_value = None

        # if the value is a text-category value
        if label in value_map:
            # if the value isn't in the list, add it
            if value not in value_map[label]:
                value_map[label].append(value)
            # get the index of the value (hence converting it into a numerical value)
            new_value = value_map[label].index(value)
        else:
            # if it is not a text-category value, leave as it is
            new_value = value

        row_to_write.append(new_value)

    # write row to file
    writer.writerow(row_to_write)

print "File has been processed."
