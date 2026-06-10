# Changelog: AI Fitness Tracker App Development

This log tracks the modifications, enhancements, and feature implementations of the Fitness Tracker application.

## [2026-06-10] Milestone 1 Layout Fixes: Tablet Centering & Title Re-alignment

### Added
*   **Friends Online/Offline Status Tracking**:
    *   Added `lastActive` timestamp tracking to the user profile model [UserProfile.kt](file:///c:/Users/rodri/OneDrive/Documentos/GitHub/Final_Project_FitnessTracker/AndroidApp_V_0.1/app/src/main/java/com/example/fitnesstrackerapp/logic/UserProfile.kt) with safe millisecond translation and fallback to `modifiedAt` if the field is missing.
    *   Configured [RegisterActivity.kt](file:///c:/Users/rodri/OneDrive/Documentos/GitHub/Final_Project_FitnessTracker/AndroidApp_V_0.1/app/src/main/java/com/example/fitnesstrackerapp/RegisterActivity.kt) and [EditProfileActivity.kt](file:///c:/Users/rodri/OneDrive/Documentos/GitHub/Final_Project_FitnessTracker/AndroidApp_V_0.1/app/src/main/java/com/example/fitnesstrackerapp/EditProfileActivity.kt) to initialize/update `lastActive` upon creation or saving.
    *   Configured [DashboardActivity.kt](file:///c:/Users/rodri/OneDrive/Documentos/GitHub/Final_Project_FitnessTracker/AndroidApp_V_0.1/app/src/main/java/com/example/fitnesstrackerapp/DashboardActivity.kt) to automatically update the current user's `lastActive` timestamp in Firestore when the dashboard opens.
    *   Implemented a query in `DashboardActivity.kt` that fetches the profile and active status of all added friends, displaying them dynamically at the bottom of the main hub under a new "Friends Status" section.
    *   Friends are listed with their name, code, a green online dot if active within 5 minutes, or a relative offline timestamp (e.g. "Active 2h ago", "Active >72h ago") if not. Added interactive placeholder text buttons for "Challenge" and "Stats" actions.
    *   Realigned the dashboard layout to a scrollable `Column` container, replacing the previous full-screen `LazyVerticalGrid` with a 3x2 chunked `Row` structure to enable scrolling down to check the friends status.
*   **Tablet Layout & Centering Constraints**: Wrapped all Compose activity screens (`LandingScreen`, `RegisterScreen`, `DashboardScreen`, `EditProfileScreen`, `PlaceholderScreen`, `LoaderScreen`) in parent `Box` containers with `Modifier.widthIn(max = 480.dp)` or `540.dp` to prevent horizontal stretching on wider screens (tablets) and keep elements elegantly centered.
*   **Locked Portrait Mode**: Updated [AndroidManifest.xml](file:///c:/Users/rodri/OneDrive/Documentos/GitHub/Final_Project_FitnessTracker/AndroidApp_V_0.1/app/src/main/AndroidManifest.xml) to explicitly add `android:screenOrientation="portrait"` to all 10 activities, preventing accidental landscape rotation that conflicts with camera-based body pose tracking.

*   **Landing Activity Branding Spacing & Font Scaling**: Refactored the branding section of `LandingScreen` in [LandingActivity.kt](file:///c:/Users/rodri/OneDrive/Documentos/GitHub/Final_Project_FitnessTracker/AndroidApp_V_0.1/app/src/main/java/com/example/fitnesstrackerapp/LandingActivity.kt):
    *   Split the "FITNESS TRACKER" branding text into two separate, explicit Compose `Text` views ("FITNESS" and "TRACKER") separated by a `6.dp` vertical `Spacer` to completely resolve vertical overlapping caused by text wrapping.
    *   Increased the branding font size from `38.sp` to `44.sp` for a more premium visual layout.
    *   Increased vertical spacing between the title branding and the subtitle ("Train anywhere you want!") to `16.dp` to prevent visual clutter.
    *   Added a `Spacer` at the top of the scrollable column to push the text block slightly downwards while keeping the entire view balanced and centered.

---

## [2026-06-10] Milestone 1 Refinements: UI/UX & Data Formatting

### Added
*   **Forgot Password Dialog**: Integrated a stateful "Forgot Password?" dialog inside [LandingActivity.kt](file:///c:/Users/rodri/OneDrive/Documentos/GitHub/Final_Project_FitnessTracker/AndroidApp_V_0.1/app/src/main/java/com/example/fitnesstrackerapp/LandingActivity.kt) to send reset verification emails using `FirebaseAuth.getInstance().sendPasswordResetEmail`.
*   **Sign Out Action**: Added a red "Sign Out" text button in [DashboardActivity.kt](file:///c:/Users/rodri/OneDrive/Documentos/GitHub/Final_Project_FitnessTracker/AndroidApp_V_0.1/app/src/main/java/com/example/fitnesstrackerapp/DashboardActivity.kt) that clears the session using `FirebaseAuth.getInstance().signOut()` and routes back to the landing page.
*   **First Name Parser**: Implemented a parser in the dashboard header that extracts and displays only the user's first name for a cleaner and more personal greeting (e.g. "Welcome Back, Rodrigo").
*   **Loader Activity**: Created [LoaderActivity.kt](file:///c:/Users/rodri/OneDrive/Documentos/GitHub/Final_Project_FitnessTracker/AndroidApp_V_0.1/app/src/main/java/com/example/fitnesstrackerapp/LoaderActivity.kt) which displays an animated, rotating neon sweep gradient spinner in the Antigravity color theme (cyan/purple) with a 1.5-second delay to ensure smooth transitions between major states.
*   **Shared Component File**: Created [PlaceholderScreen.kt](file:///c:/Users/rodri/OneDrive/Documentos/GitHub/Final_Project_FitnessTracker/AndroidApp_V_0.1/app/src/main/java/com/example/fitnesstrackerapp/PlaceholderScreen.kt) to share the "Coming Soon" screen design across the mock activities.

### Modified
*   **Prevent Screen Locking**: Added `WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON` to all activities (`LandingActivity`, `RegisterActivity`, `DashboardActivity`, `EditProfileActivity`, `CreatePlanActivity`, `StartPlanActivity`, `ViewStatisticsActivity`, and `LoaderActivity`) to prevent the screen from going to sleep while the app is active.
*   **Human-Readable Firestore Dates**: Refactored [UserProfile.kt](file:///c:/Users/rodri/OneDrive/Documentos/GitHub/Final_Project_FitnessTracker/AndroidApp_V_0.1/app/src/main/java/com/example/fitnesstrackerapp/logic/UserProfile.kt) to use native Firestore `Timestamp` objects instead of raw `Long` millisecond values. Added safe-resolution methods to parse both legacy `Long` records and new `Timestamp` objects to prevent crashes.
*   **User Numeric ID**:
    *   Updated [RegisterActivity.kt](file:///c:/Users/rodri/OneDrive/Documentos/GitHub/Final_Project_FitnessTracker/AndroidApp_V_0.1/app/src/main/java/com/example/fitnesstrackerapp/RegisterActivity.kt) to generate a random 5-digit numerical ID (e.g. `#12345`) upon account creation.
    *   Updated [DashboardActivity.kt](file:///c:/Users/rodri/OneDrive/Documentos/GitHub/Final_Project_FitnessTracker/AndroidApp_V_0.1/app/src/main/java/com/example/fitnesstrackerapp/DashboardActivity.kt) to fetch and display the user's name and 5-digit ID in the main greeting header.
*   **Friends List Search by Code**: Updated [EditProfileActivity.kt](file:///c:/Users/rodri/OneDrive/Documentos/GitHub/Final_Project_FitnessTracker/AndroidApp_V_0.1/app/src/main/java/com/example/fitnesstrackerapp/EditProfileActivity.kt) to query the Firestore `/users` collection when adding a friend by their 5-digit code, verifying their existence before adding them to the user's friends list.
*   **Transitions via Loader**:
    *   Configured [LandingActivity.kt](file:///c:/Users/rodri/OneDrive/Documentos/GitHub/Final_Project_FitnessTracker/AndroidApp_V_0.1/app/src/main/java/com/example/fitnesstrackerapp/LandingActivity.kt) to route through `LoaderActivity` to the dashboard upon login.
    *   Configured [DashboardActivity.kt](file:///c:/Users/rodri/OneDrive/Documentos/GitHub/Final_Project_FitnessTracker/AndroidApp_V_0.1/app/src/main/java/com/example/fitnesstrackerapp/DashboardActivity.kt) to route through `LoaderActivity` when launching `MainActivity` (Demo Workout) or `DemoPushUpActivity` (Demo Pushups).
*   [AndroidManifest.xml](file:///c:/Users/rodri/OneDrive/Documentos/GitHub/Final_Project_FitnessTracker/AndroidApp_V_0.1/app/src/main/AndroidManifest.xml): Registered `LoaderActivity` in the application manifest.

---

## [2026-06-10] Milestone 1: User Authentication & Cloud Database (Firebase)

### Added
*   **Firebase SDK Integration**: Added Google Services plugin and Firebase BOM to the Gradle build configuration files.
*   **User Data Model**: Created initial `UserProfile.kt` with fields for biometric data, gamification info, and social connections.
*   **Register Activity**: Created `RegisterActivity.kt` with a Compose registration form.

---

## [2026-06-10] Navigation & Jetpack Compose Base Setup

### Added
*   **Compose Support**: Integrated Jetpack Compose in the build scripts using Compose BOM `2024.10.00` and the Jetpack Compose compiler.
*   **Theme**: Created a custom dark-mode theme [Theme.kt](file:///c:/Users/rodri/OneDrive/Documentos/GitHub/Final_Project_FitnessTracker/AndroidApp_V_0.1/app/src/main/java/com/example/fitnesstrackerapp/ui/theme/Theme.kt) matching the Antigravity color palette.
*   **Landing Page Mockup**: Created initial `LandingActivity` with static buttons and form fields.
*   **Dashboard Hub**: Created `DashboardActivity.kt` to act as the primary navigation hub.
*   **Placeholders**: Created temporary activities for Edit Profile, Create Plan, Start Plan, and View Statistics.
