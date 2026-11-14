import java.util.Properties

plugins {
    id("com.android.application")
}

android {
    namespace = "com.soundwitness.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.soundwitness.app"
        minSdk = 28
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
    }

    // release signing: key and passwords live outside the repository (keystore.properties path via
    // -Psigning.props=... or the SIGNING_PROPS environment variable); without them the release build is unsigned
    val signingProps = (project.findProperty("signing.props") as String?) ?: System.getenv("SIGNING_PROPS")
    signingConfigs {
        if (signingProps != null && file(signingProps).exists()) {
            val p = Properties().apply { file(signingProps).inputStream().use { s -> load(s) } }
            create("release") {
                storeFile = file(p.getProperty("storeFile"))
                storePassword = p.getProperty("storePassword")
                keyAlias = p.getProperty("keyAlias")
                keyPassword = p.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfigs.findByName("release")?.let { signingConfig = it }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation("ai.pleos.playground:Vehicle:2.3.2")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
}

tasks.register("vehicleSdkSources") {
    val cfg = project.configurations.detachedConfiguration(
        project.dependencies.create("ai.pleos.playground:Vehicle:2.3.2:sources@jar")
    )
    doLast { cfg.files.forEach { println("SRC " + it.absolutePath) } }
}
