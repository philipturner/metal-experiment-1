//
//  Operation.swift
//  
//
//  Created by Philip Turner on 7/9/22.
//

import Metal

enum Operation {
  case increment(input: MTLBuffer, output: MTLBuffer, size: Int)
}
