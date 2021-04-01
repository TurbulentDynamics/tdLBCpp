//
//  testComputeUnit.m
//  testComputeUnit
//
//  Created by Niall Ó Broin on 16/03/2021.
//

#import <XCTest/XCTest.h>
#import "ComputeUnit.h"
#import "PlotDir.h"



@interface testComputeUnit : XCTestCase

@end

@implementation testComputeUnit

- (void)setUp {
    // Put setup code here. This method is called before the invocation of each test method in the class.
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
}

- (void)testExample {
    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
    
    GridParams grid;
    grid.x = 44;
    grid.y = 44;
    grid.z = 44;
    
    tStep num_steps = 1;

    
    grid.ngx = 1;
    grid.ngy = 1;
    grid.ngz = 1;
    
    
    FlowParams<float> flow;
    flow.initialRho = 8.0;
    flow.reMNonDimensional = 7000.0;
    flow.uav = 0.1;
    flow.g3 = 0.1;
    
    RushtonTurbine rt = RushtonTurbine(int(grid.x));
    
    Extents<tNi> e = Extents<tNi>(0, grid.x, 0, grid.y, 0, grid.z);

//    RushtonTurbineMidPointCPP<tNi> geom = RushtonTurbineMidPointCPP<tNi>(rt, e);

//    RushtonTurbinePolarCPP<tNi, useQVecPrecision> geom = RushtonTurbinePolarCPP<tNi, useQVecPrecision>(rt, e);
    ComputeUnit<float, QLen::D3Q19> lb = ComputeUnit<float, QLen::D3Q19>(1, 2, 3, 101, 102, 103, 1, flow);

    
    XCTAssertEqual(lb.idi, 1);
    
    
    std::string diskOutputDir = "diskOutputDir";
    DiskOutputTree outDir = OutputDir(diskOutputDir, grid);
    
    std::string plotPath = outDir.get_XY_plane_dir(18, 14);
    
    XCTAssertEqual(plotPath, "./diskOutputDir/plot.XYplane.V5.step00000018.cut_14");
    
    PlotDir p = PlotDir(plotPath, 1, 2, 3);

    std::string qvecPath = p.get_my_Qvec_filename(QvecNames::Qvec);
    std::cout<<qvecPath<<std::endl;
    
    XCTAssertEqual(qvecPath, "./diskOutputDir/plot.XYplane.V4.Q4.step00000018.cut14/Qvec.node.1.2.3.V5.bin");
    
    
    
}

- (void)testPerformanceExample {
    // This is an example of a performance test case.
    [self measureBlock:^{
        // Put the code you want to measure the time of here.
    }];
}

@end
