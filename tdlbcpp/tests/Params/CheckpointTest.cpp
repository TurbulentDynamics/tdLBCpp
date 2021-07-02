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
#include "Params/Checkpoint.hpp"

#include "tdlbcpp/tests/utils.hpp"

class CheckpointParamsTests : public ::testing::Test
{
protected:
    std::string filename;
    const int randomStringLength = 400;

    void checkAllFields(CheckpointParams &expected, CheckpointParams &actual)
    {
        ASSERT_EQ(expected.start_with_checkpoint, actual.start_with_checkpoint) << "start_with_checkpoint field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.load_checkpoint_dirname, actual.load_checkpoint_dirname) << "load_checkpoint_dirname field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.checkpoint_repeat, actual.checkpoint_repeat) << "checkpoint_repeat field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.checkpoint_root_dir, actual.checkpoint_root_dir) << "checkpoint_root_dir field has a wrong value after being written to a file and then read";
    }

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
    CheckpointParams checkpointParams;
    checkpointParams.start_with_checkpoint = true;
    checkpointParams.load_checkpoint_dirname = "test1";
    checkpointParams.checkpoint_repeat = 1;
    checkpointParams.checkpoint_root_dir = "test2";

    checkpointParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    CheckpointParams checkpointParamsRead;
    checkpointParamsRead.getParamsFromJsonFile(filename);

    checkAllFields(checkpointParams, checkpointParamsRead);
}

TEST_F(CheckpointParamsTests, CheckpointParamsWriteReadValidRandomTest)
{
    CheckpointParams checkpointParams;
    checkpointParams.start_with_checkpoint = (rand() & 1) == 1;
    checkpointParams.load_checkpoint_dirname = TestUtils::random_string(randomStringLength);
    checkpointParams.checkpoint_repeat = rand();
    checkpointParams.checkpoint_root_dir = TestUtils::random_string(randomStringLength);

    checkpointParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    CheckpointParams checkpointParamsRead;
    checkpointParamsRead.getParamsFromJsonFile(filename);

    checkAllFields(checkpointParams, checkpointParamsRead);
}

TEST_F(CheckpointParamsTests, CheckpointParamsReadInValidTest)
{
    std::ofstream out(filename);
    out << "{\"start_with_checkpoint\":true}";
    out.close();
    std::cerr << filename << std::endl;

    CheckpointParams checkpointParamsRead;
    checkpointParamsRead.getParamsFromJsonFile(filename);

    ASSERT_EQ(checkpointParamsRead.start_with_checkpoint, true) << "start_with_checkpoint field has a wrong value";
}
