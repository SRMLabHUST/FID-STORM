import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.frame.*;

public class FID_STORM_ extends PlugInFrame {
        public FID_STORM_()
        {
            super("FID-STORM");
            System.loadLibrary("FID-STORM");
            new Interface().setVisible(true);
        }
}
