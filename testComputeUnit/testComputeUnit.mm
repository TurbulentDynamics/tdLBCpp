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
    flow.initial_rho = 8.0;
    flow.re_m_nondimensional = 7000.0;
    flow.uav = 0.1;
    flow.g3 = 0.1;
    
    RushtonTurbine rt = RushtonTurbine(int(grid.x));
    
    Extents<tNi> e = Extents<tNi>(0, grid.x, 0, grid.y, 0, grid.z);

//    RushtonTurbineMidPointCPP<tNi> geom = RushtonTurbineMidPointCPP<tNi>(rt, e);

//    RushtonTurbinePolarCPP<tNi, usePrecision> geom = RushtonTurbinePolarCPP<tNi, usePrecision>(rt, e);
    ComputeUnit<float, QLen::D3Q19> lb = ComputeUnit<float, QLen::D3Q19>(1, 2, 3, 101, 102, 103, 1, flow);

    
    XCTAssertEqual(lb.idi, 1);
    
    
    std::string output_root_dir = "output_root_dir";
    OutputDir outDir = OutputDir(output_root_dir, grid);
    
    std::string plotPath = outDir.get_XY_plane_dir(18, 14, 4);
    
    XCTAssertEqual(plotPath, "./output_root_dir/plot.XYplane.V_4.Q_4.step_00000018.cut_14");
    
    PlotDir p = PlotDir(plotPath, 1, 2, 3);

    std::string qvecPath = p.get_my_Qvec_filename(QvecNames::Qvec);
    std::cout<<qvecPath<<std::endl;
    
    XCTAssertEqual(qvecPath, "./output_root_dir/plot.XYplane.V_4.Q_4.step_00000018.cut_14/Qvec.node.1.2.3.V4.bin");
    
    
    
}

- (void)testPerformanceExample {
    // This is an example of a performance test case.
    [self measureBlock:^{
        // Put the code you want to measure the time of here.
    }];
}

@end
