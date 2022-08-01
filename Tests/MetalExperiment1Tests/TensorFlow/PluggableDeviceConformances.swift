//
//  PluggableDeviceConformances.swift
//  
//
//  Created by Philip Turner on 7/31/22.
//

#if canImport(MetalExperiment1)
import MetalExperiment1

extension MetalExperiment1.Context: PluggableDevice {
  
}
#endif
