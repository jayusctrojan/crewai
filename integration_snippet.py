# Add this import near the top of main.py, after your existing imports
try:
    from archon_integration import setup_archon_integration
    ARCHON_INTEGRATION_AVAILABLE = True
    print("SUCCESS: Archon integration imported successfully")
except Exception as e:
    print(f"WARNING: Could not import Archon integration: {e}")
    ARCHON_INTEGRATION_AVAILABLE = False

# Add this after your app creation, around line where you initialize other components
if ARCHON_INTEGRATION_AVAILABLE:
    try:
        archon_integrator = setup_archon_integration(app)
        if archon_integrator:
            print("🏛️ Archon MCP integration activated")
    except Exception as e:
        print(f"Failed to activate Archon integration: {e}")
        archon_integrator = None
else:
    archon_integrator = None