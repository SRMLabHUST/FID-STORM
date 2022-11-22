import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.frame.*;

public class mEDSR_STORM_ extends PlugInFrame {
        public mEDSR_STORM_()
        {
            super("mEDSR_STORM");
            new Interface().setVisible(true);
        }
}
