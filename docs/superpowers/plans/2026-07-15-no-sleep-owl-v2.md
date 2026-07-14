# 不休眠猫头鹰 v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将合盖守夜升级为一次批准的特权辅助程序，并加入电源、电量和热状态自动保护。

**Architecture:** Swift Package 同时构建菜单栏主应用与 root LaunchDaemon。双方通过固定 NSXPC 接口通信；纯 Swift 策略状态机独立于系统 API，便于覆盖心跳、电池、电源和热状态测试。

**Tech Stack:** Swift 6.2、AppKit、IOKit、ServiceManagement、NSXPCConnection、launchd、macOS 15 arm64

## Global Constraints

- 目标系统 macOS 15，产物只包含 arm64。
- 特权辅助程序只允许 `enable`、`disable`、`heartbeat`、`status` 四类固定操作。
- 辅助程序 15 秒无心跳恢复原始 `SleepDisabled`。
- 默认仅接通电源时允许🦉；电池 20%提醒、10%退出；热状态 serious 提醒、critical 退出。
- 无稳定 Developer ID 的本机环境先验证 ad-hoc 注册能力；系统拒绝时如实停止，不修改 macOS 安全策略。

---

### Task 1: 安全策略状态机

**Files:**
- Create: `NoSleepOwl/Sources/NoSleepOwlCore/SafetyPolicy.swift`
- Modify: `NoSleepOwl/Tests/NoSleepOwlCoreTests/main.swift`

**Interfaces:**
- Produces: `PowerPolicy`、`SafetySnapshot`、`SafetyDecision`、`SafetyPolicy.evaluate(_:)`。

- [ ] 写拔电、20%、10%、serious、critical 的失败测试。
- [ ] 运行 `swift run NoSleepOwlTests`，确认因类型缺失失败。
- [ ] 实现纯函数策略与单次提醒状态。
- [ ] 运行测试，确认全部通过。
- [ ] 提交 `feat: add owl safety policy`。

### Task 2: 心跳安全状态机与 XPC 合约

**Files:**
- Create: `NoSleepOwl/Sources/NoSleepOwlCore/HelperStateMachine.swift`
- Create: `NoSleepOwl/Sources/NoSleepOwlCore/NoSleepOwlXPCProtocol.swift`
- Modify: `NoSleepOwl/Tests/NoSleepOwlCoreTests/main.swift`

**Interfaces:**
- Produces: `HelperStateMachine.enable(now:originalValue:)`、`heartbeat(now:)`、`tick(now:)`、`disable()`；`@objc NoSleepOwlXPCProtocol`。

- [ ] 写 15 秒超时、心跳续期、禁用恢复的失败测试。
- [ ] 运行测试，确认 RED。
- [ ] 实现状态机与只含固定方法的 XPC 合约。
- [ ] 运行测试，确认 GREEN。
- [ ] 提交 `feat: define privileged helper safety contract`。

### Task 3: LaunchDaemon 辅助程序

**Files:**
- Modify: `NoSleepOwl/Package.swift`
- Create: `NoSleepOwl/Sources/NoSleepOwlHelper/main.swift`
- Create: `NoSleepOwl/Sources/NoSleepOwlHelper/HelperService.swift`
- Create: `NoSleepOwl/Resources/com.shiying.NoSleepOwl.helper.plist`

**Interfaces:**
- Consumes: `NoSleepOwlXPCProtocol`、`HelperStateMachine`。
- Produces: Mach service `com.shiying.NoSleepOwl.helper`。

- [ ] 写 bundle 合约检查，确认 helper/plist 尚缺失时失败。
- [ ] 实现固定 `pmset` 调用、1 秒 tick、连接失效恢复和 root UID 检查。
- [ ] 构建 helper，运行无权限单元测试与 plist 校验。
- [ ] 提交 `feat: add privileged sleep helper`。

### Task 4: 主应用接入与安全监控

**Files:**
- Create: `NoSleepOwl/Sources/NoSleepOwlApp/PrivilegedSleepController.swift`
- Create: `NoSleepOwl/Sources/NoSleepOwlApp/SafetyMonitor.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/AppDelegate.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/ControlWindowController.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlCore/OwlModeStore.swift`

**Interfaces:**
- Consumes: `SMAppService.daemon`、XPC 合约、`SafetyPolicy`。
- Produces: 一次批准流程、5 秒心跳、电源策略 UI、自动退出原因。

- [ ] 为自动退出原因和策略持久化写失败测试。
- [ ] 实现辅助程序状态、XPC 控制器与 IOKit 电池读取。
- [ ] 接入窗口和菜单状态，旧 root shell 不再作为运行路径。
- [ ] 运行全部测试与 debug build。
- [ ] 提交 `feat: connect owl app to safety helper`。

### Task 5: 打包、注册与实机闭环

**Files:**
- Modify: `NoSleepOwl/scripts/build-app.sh`
- Modify: `NoSleepOwl/Resources/Info.plist`
- Modify: `NoSleepOwl/README.md`

**Interfaces:**
- Produces: 含 `Contents/Library/LaunchDaemons` 和 helper executable 的 `不休眠猫头鹰.app`。

- [ ] 更新 bundle 结构、版本号和签名顺序。
- [ ] 运行测试、release build、plist、codesign 和 arm64 检查。
- [ ] 安装 app，注册辅助程序并读取 `SMAppService.Status`。
- [ ] 首次批准后连续切换三次，确认不再询问密码。
- [ ] 🦉时确认 `SleepDisabled=1`；断开主应用后 15 秒内确认恢复为 0。
- [ ] 验证电源/电量/热策略的模拟路径。
- [ ] 提交 `build: package no-sleep owl v2 helper`。
