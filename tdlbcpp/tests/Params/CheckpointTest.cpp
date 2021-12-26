//
//  CheckpointTest.cpp
//  CheckpointTest
//
//  Unit tests for Checkpoint
//

#include <cstdio>
#include <cstdlib>

#include "gtest/gtest.h"

#include "Header.h"
#include "Params/CheckpointParams.hpp"

#include "tdlbcpp/tests/utils.hpp"
#include "tdlbcpp/tests/Params/ParamsCommon.hpp"

class CheckpointParamsTests : public ::testing::Test
{
protected:
    std::string filename;

public:
    CheckpointParamsTests()
    {
        filename = TestUtils::getTempFilename();
    }
    ~CheckpointParamsTests()
    {
        TestUtils::removeTempFile(filename);
    }
};

TEST_F(CheckpointParamsTests, CheckpointWriteReadValidTest)
{
    CheckpointParams checkpointParams = ParamsCommon::createCheckpointParamsFixed();

    checkpointParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    CheckpointParams checkpointParamsRead;
    checkpointParamsRead.getParamsFromJsonFile(filename);

    ParamsCommon::checkAllFields(checkpointParams, checkpointParamsRead);
}

TEST_F(CheckpointParamsTests, CheckpointParamsWriteReadValidRandomTest)
{
    CheckpointParams checkpointParams = ParamsCommon::createCheckpointParamsRandom();

    checkpointParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    CheckpointParams checkpointParamsRead;
    checkpointParamsRead.getParamsFromJsonFile(filename);

    ParamsCommon::checkAllFields(checkpointParams, checkpointParamsRead);
}

TEST_F(CheckpointParamsTests, CheckpointParamsReadInValidTest)
{
    std::ofstream out(filename);
    out << "{\"load_checkpoint_dirname\":\"somepath\"";
    out.close();
    std::cerr << filename << std::endl;

    CheckpointParams checkpointParamsRead;
    EXPECT_EXIT(checkpointParamsRead.getParamsFromJsonFile(filename), testing::ExitedWithCode(1), 
    "Exception reading from input file: \\* Line 1, Column 38\n"
                              "  Missing ',' or '}' in object declaration\n");
}

TEST_F(CheckpointParamsTests, CheckpointParamsReadInValidTestInvalidType)
{
    std::ofstream out(filename);
    out << "{\"load_checkpoint_dirname\":\"somepath\", \"checkpointRepeat\": \"invalidNumber\"}";
    out.close();
    std::cerr << filename << std::endl;

    CheckpointParams checkpointParamsRead;
    EXPECT_EXIT(checkpointParamsRead.getParamsFromJsonFile(filename), testing::ExitedWithCode(1), 
    "Exception reached parsing arguments in CheckpointParams: Value is not convertible to Int.\n");
}