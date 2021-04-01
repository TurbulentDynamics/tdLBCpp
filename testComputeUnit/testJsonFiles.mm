//
//  testComputeUnit.m
//  testComputeUnit
//
//  Created by Niall Ó Broin on 16/03/2021.
//

#import <XCTest/XCTest.h>
#import "ComputeUnit.h"
#import "PlotDir.h"



@interface testJsonFiles : XCTestCase

@end

@implementation testJsonFiles

- (void)setUp {
    // Put setup code here. This method is called before the invocation of each test method in the class.

    
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
}

- (void)testExample {
    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
    
    GridParams gridSave = GridParams{44,44,44,1,1,1};

    FlowParams<float> flow;
    flow.initialRho = 8.0;
    flow.reMNonDimensional = 7000.0;
    flow.uav = 0.1;
    flow.g3 = 0.1;
    
    
    gridSave.writeParams("testJson");
    
    GridParams gridLoad = grid.getParamFromJson("testJson");
    
    XCTAssertEqual(gridSave.ngx, gridLoad.ngx);
    
    
    
}

- (void)testPerformanceExample {
    // This is an example of a performance test case.
    [self measureBlock:^{
        // Put the code you want to measure the time of here.
    }];
}

@end
