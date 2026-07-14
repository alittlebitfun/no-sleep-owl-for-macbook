# 不休眠猫头鹰 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建并启动一款 Apple 芯片原生的 macOS 菜单栏应用，通过鸟类图标一键控制系统空闲休眠。

**Architecture:** 使用 Swift Package Manager 构建 AppKit 原生可执行程序，再组装为标准 `.app`。`OwlModeStore` 负责状态机，`IOKitSleepAssertionController` 负责系统电源 assertion，菜单栏控制器与窗口控制器订阅同一状态源。

**Tech Stack:** Swift 6.2、AppKit、IOKit、ServiceManagement、Swift Testing、macOS 15 arm64

## Global Constraints

- 应用名称为「不休眠猫头鹰」，Bundle ID 为 `com.shiying.NoSleepOwl`。
- 只阻止系统空闲休眠，允许显示器按系统设置熄灭。
- 首次启动和普通重新启动均进入小鸟模式；退出时释放 assertion。
- 左键切换模式，右键打开快捷菜单；应用不显示 Dock 图标。
- 不需要管理员权限，不访问网络，不收集数据。
- 发布产物必须为 arm64，并在当前 Apple 芯片 Mac 上完成自动化和人工冒烟测试。

---

### Task 1: 可测试的模式状态机与电源控制

**Files:**
- Create: `NoSleepOwl/Package.swift`
- Create: `NoSleepOwl/Sources/NoSleepOwlCore/SleepAssertionControlling.swift`
- Create: `NoSleepOwl/Sources/NoSleepOwlCore/OwlModeStore.swift`
- Create: `NoSleepOwl/Tests/NoSleepOwlCoreTests/OwlModeStoreTests.swift`

**Interfaces:**
- Produces: `SleepAssertionControlling.acquire() throws -> UInt32`、`release(_:) throws`；`@MainActor OwlModeStore.toggle()`、`shutdown()`、`mode`、`startedAt`、`errorMessage`。

- [ ] **Step 1: 写失败测试**：用内存 fake 验证开启成功、开启失败、关闭成功、释放失败、计时起点和退出清理。
- [ ] **Step 2: 验证 RED**：运行 `cd NoSleepOwl && swift test`，预期因核心类型尚不存在而失败。
- [ ] **Step 3: 最小实现**：实现 `OwlModeStore` 两态状态机；只有 acquire/release 成功后才更新状态。
- [ ] **Step 4: 验证 GREEN**：运行 `cd NoSleepOwl && swift test`，预期全部通过且 0 failures。
- [ ] **Step 5: 提交**：`git add NoSleepOwl && git commit -m "feat: add owl mode state machine"`。

### Task 2: IOKit 系统空闲休眠 assertion

**Files:**
- Create: `NoSleepOwl/Sources/NoSleepOwlCore/IOKitSleepAssertionController.swift`
- Create: `NoSleepOwl/Tests/NoSleepOwlCoreTests/IOKitSleepAssertionControllerTests.swift`

**Interfaces:**
- Consumes: `SleepAssertionControlling`。
- Produces: `IOKitSleepAssertionController`，使用 `kIOPMAssertionTypePreventUserIdleSystemSleep`，不使用显示器休眠 assertion。

- [ ] **Step 1: 写失败测试**：验证 assertion 类型选择器严格等于 `PreventUserIdleSystemSleep`。
- [ ] **Step 2: 验证 RED**：运行对应测试，预期缺少实现而失败。
- [ ] **Step 3: 最小实现**：封装 `IOPMAssertionCreateWithName` 和 `IOPMAssertionRelease`，非 success 返回可读错误。
- [ ] **Step 4: 验证 GREEN**：运行全部测试并确认通过。
- [ ] **Step 5: 提交**：`git commit -m "feat: prevent idle system sleep with IOKit"`。

### Task 3: 菜单栏左右键与状态同步

**Files:**
- Create: `NoSleepOwl/Sources/NoSleepOwlApp/StatusItemController.swift`
- Create: `NoSleepOwl/Sources/NoSleepOwlApp/BirdIconRenderer.swift`
- Create: `NoSleepOwl/Tests/NoSleepOwlCoreTests/BirdPresentationTests.swift`

**Interfaces:**
- Consumes: `OwlModeStore.toggle()`、`mode`。
- Produces: `BirdPresentation` 的图标名称、状态标题和切换标题；`StatusItemController` 的左键 toggle 与右键 menu。

- [ ] **Step 1: 写失败测试**：断言两种模式各自的图标标识、状态标题和下一操作标题。
- [ ] **Step 2: 验证 RED**：运行测试，预期 presentation 类型缺失。
- [ ] **Step 3: 最小实现**：绘制可模板化的小鸟/猫头鹰菜单栏图标，配置 `NSStatusItem` 同时接收左右键。
- [ ] **Step 4: 验证 GREEN**：运行全部测试并确认通过。
- [ ] **Step 5: 提交**：`git commit -m "feat: add two-state menu bar controls"`。

### Task 4: 控制页面与登录启动

**Files:**
- Create: `NoSleepOwl/Sources/NoSleepOwlApp/ControlWindowController.swift`
- Create: `NoSleepOwl/Sources/NoSleepOwlApp/LaunchAtLoginController.swift`
- Create: `NoSleepOwl/Sources/NoSleepOwlApp/AppDelegate.swift`
- Create: `NoSleepOwl/Sources/NoSleepOwlApp/main.swift`

**Interfaces:**
- Consumes: `OwlModeStore`、`SMAppService.mainApp`。
- Produces: 无 Dock 图标的应用生命周期、单实例控制窗口、登录启动开关、退出清理。

- [ ] **Step 1: 写失败测试**：为独立的持续时长格式函数验证 0 秒、分钟和小时显示。
- [ ] **Step 2: 验证 RED**：运行测试，预期格式函数缺失。
- [ ] **Step 3: 最小实现**：用 AppKit 构建单页窗口、模式按钮、时长、说明和登录启动开关；AppDelegate 连接所有组件。
- [ ] **Step 4: 验证 GREEN**：运行全部测试并确认通过。
- [ ] **Step 5: 提交**：`git commit -m "feat: add owl control window"`。

### Task 5: App 图标、Bundle 与本机交付

**Files:**
- Create: `NoSleepOwl/Resources/AppIcon.icns`
- Create: `NoSleepOwl/Resources/Info.plist`
- Create: `NoSleepOwl/scripts/build-app.sh`
- Create: `NoSleepOwl/README.md`
- Create: `NoSleepOwl/Tests/NoSleepOwlCoreTests/BundleContractTests.swift`

**Interfaces:**
- Consumes: release executable `NoSleepOwlApp`。
- Produces: `NoSleepOwl/dist/不休眠猫头鹰.app`。

- [ ] **Step 1: 写失败测试**：验证 Info.plist 名称、Bundle ID、agent-app 标记与 arm64 构建约束。
- [ ] **Step 2: 验证 RED**：运行测试，预期资源尚不存在。
- [ ] **Step 3: 最小实现**：生成 🦉 风格 icns，编写打包脚本，release 构建后组装 bundle 并 ad-hoc 签名。
- [ ] **Step 4: 完整验证**：依次运行 `swift test`、`scripts/build-app.sh`、`file`、`codesign --verify --deep --strict`、`plutil -lint`。
- [ ] **Step 5: 运行态验证**：启动 app，确认进程存在；切换猫头鹰模式后用 `pmset -g assertions` 确认持有 `PreventUserIdleSystemSleep`，切回后确认释放；验证右键菜单和控制窗口可打开。
- [ ] **Step 6: 安装启动**：复制到 `/Applications/不休眠猫头鹰.app`，再次验证签名和架构，然后启动供用户使用。
- [ ] **Step 7: 提交**：`git commit -m "build: package no-sleep owl app"`。
