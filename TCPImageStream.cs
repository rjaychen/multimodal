using UnityEngine;
using System.Threading;
using System.Net.Sockets;
using System.IO;
using System.Collections.Concurrent;
using System.Text;
using System.Net;

public class TCPImageStream : MonoBehaviour
{
    [SerializeField]
    private string ipAddress = "127.0.0.1";
    [SerializeField]
    private int ipPort = 8083;
    [SerializeField]
    GameObject trackedObject;
    [SerializeField]
    private int spatialMapping = 31;
    [SerializeField]
    private float defaultDepth = 5f;


    public Material sampleMaterial;
    public Texture2D tex = null;

    Thread m_NetworkThread, m_CalculationThread;
    bool m_NetworkRunning, m_Calculating = false;
    ConcurrentQueue<byte[]> dataQueue = new ConcurrentQueue<byte[]>();
    private void OnEnable()
    {
        m_NetworkRunning = true;
        m_NetworkThread = new Thread(NetworkThread);
        m_NetworkThread.Start();

    }
    private void OnDisable()
    {
        m_NetworkRunning = false;
        if (m_NetworkThread != null)
        {
            if (!m_NetworkThread.Join(100))
            {
                m_NetworkThread.Abort();
            }
        }
    }

    private void NetworkThread()
    {
        TcpClient client = new TcpClient();
        client.Connect(ipAddress, ipPort);
        Debug.Log($"Connecting to {ipAddress}/{ipPort}");
        using (var stream = client.GetStream())
        {
            BinaryReader reader = new BinaryReader(stream);
            try
            {
                while (m_NetworkRunning && client.Connected && stream.CanRead)
                {
                    //int length = IPAddress.NetworkToHostOrder(reader.ReadInt32()); // get file size
                    //// Debug.Log($"received filesize: {length}");
                    //byte[] data = reader.ReadBytes(length); // get data
                    //// Debug.Log($"read {length} bytes");
                    //dataQueue.Enqueue(data);

                    // Read the total length of the data (big-endian int32)
                    int totalLength = IPAddress.NetworkToHostOrder(reader.ReadInt32());

                    // Read the bbox data (6 integers)
                    int bbox_x = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                    int bbox_y = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                    int bbox_w = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                    int bbox_h = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                    int height = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                    int width = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                    if (!m_Calculating) {
                        m_CalculationThread = new Thread(() => ImageToWorld((float)bbox_x, (float)bbox_y,
                                                                            (float)bbox_w, (float)bbox_h,
                                                                            (float)height, (float)width));
                        m_CalculationThread.Start();
                    }
                    
                    // Read the image data
                    byte[] imageData = reader.ReadBytes(totalLength - 24); // Subtract 24 bytes for the bbox data

                    dataQueue.Enqueue(imageData); // Enqueue the data
                }
            }
            catch { }
        }
    }

    private void ImageToWorld(float bbox_x, float bbox_y, float bbox_w, float bbox_h, float height, float width)
    {
        m_Calculating = true;
        Debug.Log($"{bbox_x}, {bbox_y}, {bbox_w}, {bbox_h}, {height}, {width}");

        // Image Space to Screen Space 
        Vector2[] screenCoords = new Vector2[4];

        screenCoords[0] = new Vector2(bbox_x / width, bbox_y / height); // top left
        screenCoords[1] = new Vector2((bbox_x + bbox_w) / width, bbox_y / height);
        screenCoords[2] = new Vector2(bbox_x / width, (bbox_y + bbox_h) / height); // bottom left
        screenCoords[3] = new Vector2((bbox_x + bbox_w) / width, (bbox_y + bbox_h) / height);

        Debug.Log($"{screenCoords[1]}");

        // Screen to Viewport
        Vector3[] viewportCoords = new Vector3[screenCoords.Length];
        for (int i = 0; i < screenCoords.Length; i++)
        {
            viewportCoords[i] = Camera.main.ScreenToViewportPoint(screenCoords[i]);
        }
        Debug.Log(viewportCoords[3]);


        //Vector3[] worldCoords = new Vector3[viewportCoords.Length];
        //Vector3 center = Vector3.zero;
        //// // More advanced spatial mapping tracking
        ////for (int i = 0; i < viewportCoords.Length; i++)
        ////{
        ////    // Raycast to find the depth using spatial mapping
        ////    RaycastHit hit;
        ////    if (Physics.Raycast(Camera.main.ViewportPointToRay(viewportCoords[i]), out hit, Mathf.Infinity, spatialMapping))
        ////    {
        ////        worldCoords[i] = hit.point;
        ////    }
        ////    else
        ////    {
        ////        // Fallback: Use a default depth
        ////        worldCoords[i] = Camera.main.ViewportToWorldPoint(new Vector3(viewportCoords[i].x, viewportCoords[i].y, defaultDepth));
        ////    }
        ////    center += worldCoords[i];
        ////}

        //for (int i = 0; i < viewportCoords.Length; i++)
        //{
        //    // Assuming Z value is the same as the distance from the camera
        //    worldCoords[i] = Camera.main.ViewportToWorldPoint(new Vector3(viewportCoords[i].x, viewportCoords[i].y, 0.5f));
        //    center += worldCoords[i];
        //}
        //float quadWidth = Mathf.Abs(worldCoords[1].x - worldCoords[0].x);
        //float quadHeight = Mathf.Abs(worldCoords[2].y - worldCoords[0].y);
        //center /= worldCoords.Length;

        //Debug.Log(center);
        //trackedObject.transform.position = center;
        //trackedObject.transform.localScale = new Vector3(quadWidth, quadHeight, 1f);
        m_Calculating = false;
    }

    void Update()
    {
        if (dataQueue.Count > 0 && dataQueue.TryDequeue(out byte[] data))
        {
            if (tex == null)
                tex = new Texture2D(1, 1);
            bool isLoaded = tex.LoadImage(data);
            if (isLoaded)
            {
                tex.Apply();
                sampleMaterial.mainTexture = tex;
            }
            else
            {
                Debug.LogError("Failed to load image data into texture.");
            }
        }
    }
}