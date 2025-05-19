class FolderInfoIterator:
    def __init__(self, folder_info_list):
        self.folder_info_list = folder_info_list
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.folder_info_list):
            item = self.folder_info_list[self.index]
            self.index += 1
            return item
        else:
            raise StopIteration

    def get_by_index(self, index):
        """Retrieves the item at the specified index."""
        if 0 <= index < len(self.folder_info_list):
            return self.folder_info_list[index]
        else:
            raise IndexError(f"Index {index} is out of bounds.")

    def get_by_attribute(self, attribute_name, attribute_value):
        """Retrieves the first item where the specified attribute matches the given value."""
        for item in self.folder_info_list:
            if item.get(attribute_name) == attribute_value:
                return item
        return None  # Return None if no matching item is found


# Create an instance of the iterator
folder_info_iterator = FolderInfoIterator(input_folder_info)

# Get item by index
print("Getting item by index:")
try:
    item_at_index_1 = folder_info_iterator.get_by_index(1)
    print(f"Item at index 1: {item_at_index_1}")
except IndexError as e:
    print(e)

try:
    item_at_index_4 = folder_info_iterator.get_by_index(4)
    print(f"Item at index 4: {item_at_index_4}")
except IndexError as e:
    print(e)

print("-" * 20)

# Get item by vector_output_name
print("\nGetting item by vector_output_name:")
item_by_name_canada = folder_info_iterator.get_by_attribute("vector_output_name", "Canada")
print(f"Item with vector_output_name 'Canada': {item_by_name_canada}")

item_by_name_vedic = folder_info_iterator.get_by_attribute("vector_output_name", "VedicMetaverses")
print(f"Item with vector_output_name 'VedicMetaverses': {item_by_name_vedic}")

item_by_name_nonexistent = folder_info_iterator.get_by_attribute("vector_output_name", "NonExistent")
print(f"Item with vector_output_name 'NonExistent': {item_by_name_nonexistent}")

print("-" * 20)

# You can still iterate through all items
print("\nIterating through all items:")
for item in folder_info_iterator:
    print(item)

# Note: After the for loop, the internal index of the iterator will be at the end.
# If you want to iterate again, you'd need to reset it:
folder_info_iterator.index = 0