class DeviceManager:
    """Stores the device being used for calculations."""
    def __init__(self, device):
        self.device = device

_DEVICE_CONTEXT = DeviceManager('cpu')
