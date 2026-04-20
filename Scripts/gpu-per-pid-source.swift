import IOKit
import Foundation

// Returns (pid, processName, accumulatedGPUTimeNs) for all GPU clients
func getAllGPUClientStats() -> [(Int, String, UInt64)] {
    var results: [(Int, String, UInt64)] = []
    var iterator: io_iterator_t = 0
    let matching = IOServiceMatching("AGXAcceleratorG16G") ?? IOServiceMatching("IOAccelerator")
    IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator)
    let parent = IOIteratorNext(iterator)
    guard parent != 0 else { return results }
    
    var childIter: io_iterator_t = 0
    IORegistryEntryGetChildIterator(parent, kIOServicePlane, &childIter)
    var child = IOIteratorNext(childIter)
    
    while child != 0 {
        var props: Unmanaged<CFMutableDictionary>?
        IORegistryEntryCreateCFProperties(child, &props, kCFAllocatorDefault, 0)
        if let dict = props?.takeRetainedValue() as? [String: Any],
           let creator = dict["IOUserClientCreator"] as? String,
           let appUsage = dict["AppUsage"] as? [[String: Any]] {
            let parts = creator.components(separatedBy: ", ")
            if let pidStr = parts.first?.dropFirst(4), let pid = Int(pidStr) {
                let name = parts.count > 1 ? String(parts[1]) : "unknown"
                var totalGPUNs: UInt64 = 0
                for usage in appUsage {
                    if let gpuTime = usage["accumulatedGPUTime"] as? UInt64 {
                        totalGPUNs += gpuTime
                    } else if let gpuTime = usage["accumulatedGPUTime"] as? Int {
                        totalGPUNs += UInt64(gpuTime)
                    }
                }
                results.append((pid, name, totalGPUNs))
            }
        }
        IOObjectRelease(child)
        child = IOIteratorNext(childIter)
    }
    IOObjectRelease(childIter)
    IOObjectRelease(parent)
    IOObjectRelease(iterator)
    return results
}

// Output as JSON for easy parsing
let stats = getAllGPUClientStats()
var jsonArray: [[String: Any]] = []
for (pid, name, gpuNs) in stats {
    jsonArray.append(["pid": pid, "name": name, "gpu_ns": gpuNs])
}
if let data = try? JSONSerialization.data(withJSONObject: jsonArray),
   let str = String(data: data, encoding: .utf8) {
    print(str)
}
