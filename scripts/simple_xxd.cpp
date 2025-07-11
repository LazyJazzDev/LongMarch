#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Function to replace illegal characters in a variable name
std::string sanitizeVariableName(const std::string &fileName) {
  std::string sanitizedName = fileName;
  for (char &ch : sanitizedName) {
    if (ch == '.' || ch == '/' || ch == '\\') {
      ch = '_';
    }
  }
  return sanitizedName;
}

// Function to read the file into a vector of bytes
std::vector<unsigned char> readFile(const std::string &filePath) {
  std::ifstream file(filePath, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Error: Unable to open input file: " + filePath);
  }
  return std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
}

// Function to write the C++ array to the output file
void writeOutputFile(const std::string &outFilePath,
                     const std::string &varName,
                     const std::vector<unsigned char> &fileContent) {
  std::ofstream outFile(outFilePath);
  if (!outFile) {
    throw std::runtime_error("Error: Unable to open output file: " + outFilePath);
  }

  outFile << "// This file is automatically generated by simple_xxd\n";

  // Start writing the C++ array
  outFile << "unsigned char " << varName << "[] = { ";

  // Write the file content as hexadecimal values
  for (size_t i = 0; i < fileContent.size(); ++i) {
    if (i > 0) {
      outFile << ", ";
    }
    outFile << "0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(fileContent[i]);
  }
  outFile << " };\n";

  // Write the length of the array
  outFile << "unsigned int " << varName << "_len = " << std::to_string(fileContent.size()) << ";\n";
  outFile.close();
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input file path> <output file path>\n";
    return 1;
  }

  try {
    std::string inputFilePath = argv[1];
    std::string outputFilePath = argv[2];

    // Output debugging information
    std::cout << "Current working directory: " << std::filesystem::current_path() << "\n";
    std::cout << "Input file path: " << inputFilePath << "\n";
    std::cout << "Output file path: " << outputFilePath << "\n";

    // Extract file name and sanitize it
    std::string fileName = inputFilePath;  // std::filesystem::path(inputFilePath).filename().string();
    std::string varName = sanitizeVariableName(fileName);

    std::cout << "Variable name: " << varName << "\n";

    // Read the file content
    std::vector<unsigned char> fileContent = readFile(inputFilePath);

    std::cout << "Input file path: " << inputFilePath << "\n";
    std::cout << "Output file path: " << outputFilePath << "\n";

    // Write the output file
    writeOutputFile(outputFilePath, varName, fileContent);

    std::cout << "Successfully written to output file.\n";
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }

  return 0;
}
