from .base_command import BaseCommand
import os
import shutil
from typing import List

class CopyCommand(BaseCommand):
    def __init__(self, options: List[str], args: List[str]) -> None:
        """
        Initialize the CopyCommand object.

        Args:
            options (List[str]): List of command options.
            args (List[str]): List of command arguments.
        """
        super().__init__(options, args)

        # Override the attributes inherited from BaseCommand
        self.description = 'Copy a file or directory to another location'
        self.usage = 'Usage: cp [source] [destination]'
        
        self.name = 'cp'
        self.options = options
        self.src_dic = self.args[0]
        self.des_dic = self.args[1]

        # TODO 6-1: Initialize any additional attributes you may need.
        # Refer to list_command.py, grep_command.py to implement this.
        # ...

    def execute(self) -> None:
        """
        Execute the copy command.
        Supported options:
            -i: Prompt the user before overwriting an existing file.
            -v: Enable verbose mode (print detailed information)
        
        TODO 6-2: Implement the functionality to copy a file or directory to another location.
        You may need to handle exceptions and print relevant error messages.
        You may use the file_exists() method to check if the destination file already exists.
        """
        # Your code here
        overwrite_file = '-i' in self.options
        text_print =  '-v' in self.options
        
        if text_print:
            print(f"cp: copying '{self.src_dic}' to '{self.des_dic}'")
        
        product = ''
        if '/' in self.src_dic:
            list = self.src_dic.split('/')
            list = [word for word in list if word]
            product = list[-1]
        else:
            product = self.src_dic
            
            
        if '/' != self.des_dic[-1]:
            self.des_dic += '/'
        try:
            if self.file_exists(self.des_dic, product):
                if overwrite_file:
                    print(f"mv: overwrite '{self.des_dic}{self.src_dic}'? (y/n)")
                    command = input(": ")
                    if command == 'y':
                        re_dest = os.path.join(self.des_dic, product)
                        shutil.copy(self.src_dic, self.des_dic)
                    
                    else:
                        pass
                        
                else:
                    shutil.copy(self.src_dic, self.des_dic)
                
            else:
                shutil.copy(self.src_dic, self.des_dic)
        except FileNotFoundError:
            pass
        except:
            pass
        

    def file_exists(self, directory: str, file_name: str) -> bool:
        """
        Check if a file exists in a directory.
        Feel free to use this method in your execute() method.

        Args:
            directory (str): The directory to check.
            file_name (str): The name of the file.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        file_path = os.path.join(directory, file_name)
        return os.path.exists(file_path)
