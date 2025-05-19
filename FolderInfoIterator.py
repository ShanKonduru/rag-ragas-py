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
