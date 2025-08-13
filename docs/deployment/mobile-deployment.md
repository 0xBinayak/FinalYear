# Mobile Deployment Guide

This guide covers deploying the Advanced Federated Pipeline mobile client to Android and iOS devices, as well as cross-platform deployment strategies.

## Supported Platforms

- Android 8.0+ (API level 26+)
- iOS 12.0+
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 18.04+)

## Prerequisites

### Development Environment

**For Android:**
- Android Studio 4.0+
- Android SDK 30+
- NDK 21.0+
- Java 8+

**For iOS:**
- Xcode 12.0+
- iOS SDK 14.0+
- macOS development machine
- Apple Developer Account

**For Cross-Platform:**
- Flutter 3.0+ or React Native 0.68+
- Python 3.8+
- Docker (for containerized deployment)

## Android Deployment

### Step 1: Build Android APK

```bash
# Clone repository
git clone <repository-url>
cd advanced-federated-pipeline/mobile

# Install dependencies
flutter pub get

# Build for Android
flutter build apk --release

# Or build App Bundle for Play Store
flutter build appbundle --release
```

### Step 2: Configure Android Manifest

```xml
<!-- android/app/src/main/AndroidManifest.xml -->
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.federated.pipeline">
    
    <!-- Network permissions -->
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
    <uses-permission android:name="android.permission.ACCESS_WIFI_STATE" />
    
    <!-- Location permissions for signal geolocation -->
    <uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
    <uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
    
    <!-- Storage permissions for model caching -->
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
    
    <!-- Background processing -->
    <uses-permission android:name="android.permission.WAKE_LOCK" />
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
    
    <!-- USB permissions for SDR dongles -->
    <uses-permission android:name="android.permission.USB_PERMISSION" />
    
    <application
        android:label="Federated Pipeline"
        android:icon="@mipmap/ic_launcher"
        android:usesCleartextTraffic="false"
        android:networkSecurityConfig="@xml/network_security_config">
        
        <!-- Main activity -->
        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:launchMode="singleTop"
            android:theme="@style/LaunchTheme">
            <intent-filter>
                <action android:name="android.intent.action.MAIN"/>
                <category android:name="android.intent.category.LAUNCHER"/>
            </intent-filter>
        </activity>
        
        <!-- Background service -->
        <service
            android:name=".FederatedLearningService"
            android:enabled="true"
            android:exported="false"
            android:foregroundServiceType="dataSync" />
        
        <!-- USB device filter -->
        <activity android:name=".UsbActivity">
            <intent-filter>
                <action android:name="android.hardware.usb.action.USB_DEVICE_ATTACHED" />
            </intent-filter>
            <meta-data android:name="android.hardware.usb.action.USB_DEVICE_ATTACHED"
                android:resource="@xml/device_filter" />
        </activity>
    </application>
</manifest>
```

### Step 3: Network Security Configuration

```xml
<!-- android/app/src/main/res/xml/network_security_config.xml -->
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <domain-config cleartextTrafficPermitted="false">
        <domain includeSubdomains="true">your-federated-server.com</domain>
        <pin-set expiration="2025-12-31">
            <pin digest="SHA-256">your-certificate-pin</pin>
        </pin-set>
    </domain-config>
    
    <!-- Allow localhost for development -->
    <domain-config cleartextTrafficPermitted="true">
        <domain includeSubdomains="true">localhost</domain>
        <domain includeSubdomains="true">10.0.2.2</domain>
    </domain-config>
</network-security-config>
```

### Step 4: Build Configuration

```gradle
// android/app/build.gradle
android {
    compileSdkVersion 33
    ndkVersion "21.4.7075529"
    
    defaultConfig {
        applicationId "com.federated.pipeline"
        minSdkVersion 26
        targetSdkVersion 33
        versionCode 1
        versionName "1.0.0"
        
        ndk {
            abiFilters 'arm64-v8a', 'armeabi-v7a', 'x86_64'
        }
    }
    
    buildTypes {
        release {
            minifyEnabled true
            shrinkResources true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
            
            buildConfigField "String", "SERVER_URL", '"https://your-federated-server.com"'
            buildConfigField "boolean", "DEBUG_MODE", "false"
        }
        
        debug {
            buildConfigField "String", "SERVER_URL", '"http://10.0.2.2:8000"'
            buildConfigField "boolean", "DEBUG_MODE", "true"
        }
    }
    
    packagingOptions {
        pickFirst '**/libc++_shared.so'
        pickFirst '**/libjsc.so'
    }
}

dependencies {
    implementation 'androidx.work:work-runtime:2.8.1'
    implementation 'androidx.lifecycle:lifecycle-service:2.6.2'
    implementation 'com.squareup.retrofit2:retrofit:2.9.0'
    implementation 'com.squareup.retrofit2:converter-gson:2.9.0'
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
}
```

## iOS Deployment

### Step 1: Configure iOS Project

```bash
# Build for iOS
flutter build ios --release

# Open in Xcode
open ios/Runner.xcworkspace
```

### Step 2: Info.plist Configuration

```xml
<!-- ios/Runner/Info.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- App Transport Security -->
    <key>NSAppTransportSecurity</key>
    <dict>
        <key>NSExceptionDomains</key>
        <dict>
            <key>your-federated-server.com</key>
            <dict>
                <key>NSExceptionRequiresForwardSecrecy</key>
                <false/>
                <key>NSExceptionMinimumTLSVersion</key>
                <string>TLSv1.2</string>
            </dict>
        </dict>
    </dict>
    
    <!-- Location permissions -->
    <key>NSLocationWhenInUseUsageDescription</key>
    <string>This app needs location access to associate signal data with geographic coordinates for federated learning.</string>
    
    <key>NSLocationAlwaysAndWhenInUseUsageDescription</key>
    <string>This app needs location access to associate signal data with geographic coordinates for federated learning.</string>
    
    <!-- Background processing -->
    <key>UIBackgroundModes</key>
    <array>
        <string>background-processing</string>
        <string>background-fetch</string>
    </array>
    
    <!-- Network usage -->
    <key>NSLocalNetworkUsageDescription</key>
    <string>This app needs local network access to communicate with edge coordinators.</string>
</dict>
</plist>
```

### Step 3: Background Processing

```swift
// ios/Runner/AppDelegate.swift
import UIKit
import Flutter
import BackgroundTasks

@UIApplicationMain
@objc class AppDelegate: FlutterAppDelegate {
    override func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        
        // Register background tasks
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: "com.federated.pipeline.training",
            using: nil
        ) { task in
            self.handleBackgroundTraining(task: task as! BGProcessingTask)
        }
        
        GeneratedPluginRegistrant.register(with: self)
        return super.application(application, didFinishLaunchingWithOptions: launchOptions)
    }
    
    func handleBackgroundTraining(task: BGProcessingTask) {
        // Schedule next background task
        scheduleBackgroundTraining()
        
        // Perform federated learning training
        let operation = FederatedTrainingOperation()
        
        task.expirationHandler = {
            operation.cancel()
        }
        
        operation.completionBlock = {
            task.setTaskCompleted(success: !operation.isCancelled)
        }
        
        OperationQueue().addOperation(operation)
    }
    
    func scheduleBackgroundTraining() {
        let request = BGProcessingTaskRequest(identifier: "com.federated.pipeline.training")
        request.requiresNetworkConnectivity = true
        request.requiresExternalPower = false
        request.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60) // 15 minutes
        
        try? BGTaskScheduler.shared.submit(request)
    }
}
```

## Cross-Platform Configuration

### Flutter Configuration

```yaml
# pubspec.yaml
name: federated_pipeline_mobile
description: Mobile client for Advanced Federated Pipeline

version: 1.0.0+1

environment:
  sdk: ">=2.17.0 <4.0.0"
  flutter: ">=3.0.0"

dependencies:
  flutter:
    sdk: flutter
  
  # Networking
  http: ^0.13.5
  dio: ^5.3.2
  connectivity_plus: ^4.0.2
  
  # Machine Learning
  tflite_flutter: ^0.10.4
  tflite_flutter_helper: ^0.3.1
  
  # Storage
  sqflite: ^2.3.0
  shared_preferences: ^2.2.2
  path_provider: ^2.1.1
  
  # Background processing
  workmanager: ^0.5.2
  background_fetch: ^1.3.6
  
  # Device info
  device_info_plus: ^9.1.0
  battery_plus: ^4.0.2
  
  # Location
  geolocator: ^9.0.2
  
  # Permissions
  permission_handler: ^11.0.1
  
  # Crypto
  crypto: ^3.0.3
  encrypt: ^5.0.1
  
  # UI
  flutter_bloc: ^8.1.3
  get_it: ^7.6.4

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^2.0.3
  build_runner: ^2.4.7

flutter:
  uses-material-design: true
  
  assets:
    - assets/models/
    - assets/config/
    - assets/images/
```

### Main Application Structure

```dart
// lib/main.dart
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:workmanager/workmanager.dart';
import 'package:get_it/get_it.dart';

import 'core/di/injection.dart';
import 'core/services/federated_service.dart';
import 'presentation/app.dart';

void callbackDispatcher() {
  Workmanager().executeTask((task, inputData) async {
    switch (task) {
      case 'federatedTraining':
        final service = GetIt.instance<FederatedService>();
        await service.performBackgroundTraining();
        break;
      case 'modelSync':
        final service = GetIt.instance<FederatedService>();
        await service.syncWithServer();
        break;
    }
    return Future.value(true);
  });
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize dependency injection
  await configureDependencies();
  
  // Initialize background work manager
  await Workmanager().initialize(callbackDispatcher, isInDebugMode: false);
  
  // Schedule periodic tasks
  await Workmanager().registerPeriodicTask(
    "federatedTraining",
    "federatedTraining",
    frequency: Duration(hours: 1),
    constraints: Constraints(
      networkType: NetworkType.connected,
      requiresBatteryNotLow: true,
    ),
  );
  
  runApp(FederatedPipelineApp());
}
```

### Configuration Management

```dart
// lib/core/config/app_config.dart
class AppConfig {
  static const String serverUrl = String.fromEnvironment(
    'SERVER_URL',
    defaultValue: 'https://federated-server.com',
  );
  
  static const bool debugMode = bool.fromEnvironment(
    'DEBUG_MODE',
    defaultValue: false,
  );
  
  static const int maxRetries = 3;
  static const Duration connectionTimeout = Duration(seconds: 30);
  static const Duration trainingInterval = Duration(hours: 1);
  static const double minBatteryLevel = 0.3;
  static const int maxModelSize = 50 * 1024 * 1024; // 50MB
}
```

## Device-Specific Optimizations

### Battery Optimization

```dart
// lib/core/services/battery_service.dart
import 'package:battery_plus/battery_plus.dart';

class BatteryService {
  final Battery _battery = Battery();
  
  Future<bool> canPerformTraining() async {
    final batteryLevel = await _battery.batteryLevel;
    final batteryState = await _battery.batteryState;
    
    // Only train if battery > 30% or charging
    return batteryLevel > 30 || batteryState == BatteryState.charging;
  }
  
  Future<void> optimizeForBattery() async {
    final batteryLevel = await _battery.batteryLevel;
    
    if (batteryLevel < 50) {
      // Reduce training frequency
      await Workmanager().cancelByUniqueName('federatedTraining');
      await Workmanager().registerPeriodicTask(
        "federatedTraining",
        "federatedTraining",
        frequency: Duration(hours: 2), // Reduced frequency
      );
    }
  }
}
```

### Network Optimization

```dart
// lib/core/services/network_service.dart
import 'package:connectivity_plus/connectivity_plus.dart';

class NetworkService {
  final Connectivity _connectivity = Connectivity();
  
  Future<bool> isOptimalForTraining() async {
    final connectivityResult = await _connectivity.checkConnectivity();
    
    switch (connectivityResult) {
      case ConnectivityResult.wifi:
        return true; // WiFi is optimal
      case ConnectivityResult.mobile:
        return await _isMobileDataSufficient();
      default:
        return false;
    }
  }
  
  Future<bool> _isMobileDataSufficient() async {
    // Check if user allows mobile data usage
    // Check data usage limits
    // Return based on user preferences
    return false; // Conservative default
  }
  
  Future<void> adaptToNetworkConditions() async {
    final connectivityResult = await _connectivity.checkConnectivity();
    
    if (connectivityResult == ConnectivityResult.mobile) {
      // Use compression for mobile networks
      // Reduce model update frequency
      // Use differential updates only
    }
  }
}
```

## Security Configuration

### Certificate Pinning

```dart
// lib/core/network/certificate_pinning.dart
import 'dart:io';
import 'package:dio/dio.dart';

class CertificatePinning {
  static Dio createSecureClient() {
    final dio = Dio();
    
    (dio.httpClientAdapter as DefaultHttpClientAdapter).onHttpClientCreate = (client) {
      client.badCertificateCallback = (cert, host, port) {
        // Implement certificate pinning
        final expectedFingerprint = 'your-certificate-fingerprint';
        final actualFingerprint = cert.sha256.toString();
        
        return actualFingerprint == expectedFingerprint;
      };
      
      return client;
    };
    
    return dio;
  }
}
```

### Data Encryption

```dart
// lib/core/security/encryption_service.dart
import 'package:encrypt/encrypt.dart';

class EncryptionService {
  late final Encrypter _encrypter;
  late final IV _iv;
  
  EncryptionService() {
    final key = Key.fromSecureRandom(32);
    _encrypter = Encrypter(AES(key));
    _iv = IV.fromSecureRandom(16);
  }
  
  String encryptModelData(String data) {
    final encrypted = _encrypter.encrypt(data, iv: _iv);
    return encrypted.base64;
  }
  
  String decryptModelData(String encryptedData) {
    final encrypted = Encrypted.fromBase64(encryptedData);
    return _encrypter.decrypt(encrypted, iv: _iv);
  }
}
```

## Testing and Debugging

### Unit Testing

```dart
// test/services/federated_service_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:mockito/mockito.dart';

import '../mocks/mock_services.dart';

void main() {
  group('FederatedService', () {
    late FederatedService service;
    late MockNetworkService mockNetworkService;
    late MockBatteryService mockBatteryService;
    
    setUp(() {
      mockNetworkService = MockNetworkService();
      mockBatteryService = MockBatteryService();
      service = FederatedService(
        networkService: mockNetworkService,
        batteryService: mockBatteryService,
      );
    });
    
    test('should not train when battery is low', () async {
      when(mockBatteryService.canPerformTraining())
          .thenAnswer((_) async => false);
      
      final result = await service.shouldPerformTraining();
      
      expect(result, false);
    });
    
    test('should train when conditions are optimal', () async {
      when(mockBatteryService.canPerformTraining())
          .thenAnswer((_) async => true);
      when(mockNetworkService.isOptimalForTraining())
          .thenAnswer((_) async => true);
      
      final result = await service.shouldPerformTraining();
      
      expect(result, true);
    });
  });
}
```

### Integration Testing

```dart
// integration_test/app_test.dart
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'package:federated_pipeline_mobile/main.dart' as app;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  
  group('Federated Pipeline App', () {
    testWidgets('should connect to server and sync model', (tester) async {
      app.main();
      await tester.pumpAndSettle();
      
      // Test app initialization
      expect(find.text('Federated Pipeline'), findsOneWidget);
      
      // Test server connection
      await tester.tap(find.text('Connect'));
      await tester.pumpAndSettle();
      
      // Verify connection status
      expect(find.text('Connected'), findsOneWidget);
    });
  });
}
```

## Distribution

### Android Distribution

```bash
# Build signed APK
flutter build apk --release --split-per-abi

# Upload to Play Store
# Use Play Console or fastlane for automation
```

### iOS Distribution

```bash
# Build for App Store
flutter build ios --release

# Archive and upload using Xcode or fastlane
```

### Enterprise Distribution

```yaml
# fastlane/Fastfile
default_platform(:android)

platform :android do
  desc "Deploy to internal testing"
  lane :internal do
    gradle(task: "clean assembleRelease")
    upload_to_play_store(
      track: 'internal',
      apk: 'build/app/outputs/flutter-apk/app-release.apk'
    )
  end
end

platform :ios do
  desc "Deploy to TestFlight"
  lane :beta do
    build_app(scheme: "Runner")
    upload_to_testflight
  end
end
```

This completes the mobile deployment guide with comprehensive coverage of Android, iOS, and cross-platform deployment strategies, including security, optimization, and distribution.