//
//  BinFileTest.cpp
//  BinFileTest
//
//
//

#include "gtest/gtest.h"

#include "Header.h"
#include "Params/BinFile.hpp"

TEST(JsonTests, BinFileValidTest)
{
    BinFileFormat binFileFormat;
    binFileFormat.filePath = "test";
    binFileFormat.name = "test1";
    binFileFormat.note = "test2";
    binFileFormat.structName = "test3";
    binFileFormat.binFileSizeInStructs = 2;
    binFileFormat.coordsType = "test4";

    char *fname = "testfile_to_delete.json";
    binFileFormat.writeParams(fname);

    BinFileFormat binFileFormatToRead;
    binFileFormatToRead.getParamFromJson(fname);

	ASSERT_EQ(binFileFormatToRead.filePath, binFileFormat.filePath) << "File path not equal";
}










