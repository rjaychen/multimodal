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
    private LayerMask spatialMask;
    [SerializeField]
    private float defaultDepth = 4f;


    public Material sampleMaterial;
    public Texture2D tex = null;

    Thread m_NetworkThread;
    bool m_NetworkRunning;
    ConcurrentQueue<byte[]> imageQueue = new ConcurrentQueue<byte[]>();
    ConcurrentQueue<Vector2[]> bboxQueue = new ConcurrentQueue<Vector2[]>();
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
                    //imageQueue.Enqueue(data);

                    // Read the total length of the data (big-endian int32)
                    int totalLength = IPAddress.NetworkToHostOrder(reader.ReadInt32());

                    // Read the bbox data (6 integers)
                    int bbox_x = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                    int bbox_y = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                    int bbox_w = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                    int bbox_h = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                    int height = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                    int width = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                    Vector2[] screenCoords = ImageToViewport((float)bbox_x, (float)bbox_y,
                                                           (float)bbox_w, (float)bbox_h,
                                                           (float)height, (float)width);
                    bboxQueue.Enqueue(screenCoords);
                    
                    // Read the image data
                    byte[] imageData = reader.ReadBytes(totalLength - 24); // Subtract 24 bytes for the bbox data

                    imageQueue.Enqueue(imageData); // Enqueue the data
                }
            }
            catch { }
        }
    }

    private Vector2[] ImageToViewport(float bbox_x, float bbox_y, float bbox_w, float bbox_h, float height, float width)
    {
        // Debug.Log($"{bbox_x}, {bbox_y}, {bbox_w}, {bbox_h}, {height}, {width}");

        // Image Space to Screen Space 
        Vector2[] viewportCoords = new Vector2[4];

        viewportCoords[0] = new Vector2(bbox_x / width, bbox_y / height); // top left
        viewportCoords[1] = new Vector2((bbox_x + bbox_w) / width, bbox_y / height);
        viewportCoords[2] = new Vector2(bbox_x / width, (bbox_y + bbox_h) / height); // bottom left
        viewportCoords[3] = new Vector2((bbox_x + bbox_w) / width, (bbox_y + bbox_h) / height);

        return viewportCoords;
    }

    void Update()
    {
        if (imageQueue.Count > 0 && imageQueue.TryDequeue(out byte[] data))
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

        if (bboxQueue.Count > 0 && bboxQueue.TryDequeue(out Vector2[] viewportCoords))
        {
            // Screen to Viewport
            Vector3 center = Vector3.zero;
            Vector3[] worldCoords = new Vector3[viewportCoords.Length];
            for (int i = 0; i < viewportCoords.Length; i++)
            {
                worldCoords[i] = Camera.main.ViewportToWorldPoint(new Vector3(viewportCoords[i].x, viewportCoords[i].y, defaultDepth));
                center += worldCoords[i];
            }
            Debug.Log($"{worldCoords[0]} | {worldCoords[1]} \n{worldCoords[2]} | {worldCoords[3]}");

            
            // // More advanced spatial mapping tracking
            //for (int i = 0; i < viewportCoords.Length; i++)
            //{
            //    // Raycast to find the depth using spatial mapping
            //    RaycastHit hit;
            //    if (Physics.Raycast(Camera.main.ViewportPointToRay(viewportCoords[i]), out hit, Mathf.Infinity, spatialMapping))
            //    {
            //        worldCoords[i] = hit.point;
            //    }
            //    else
            //    {
            //        // Fallback: Use a default depth
            //        worldCoords[i] = Camera.main.ViewportToWorldPoint(new Vector3(viewportCoords[i].x, viewportCoords[i].y, defaultDepth));
            //    }
            //    center += worldCoords[i];
            //}
            float quadWidth = Mathf.Abs(worldCoords[1].x - worldCoords[0].x);
            float quadHeight = Mathf.Abs(worldCoords[2].y - worldCoords[0].y);
            center /= worldCoords.Length;

            // Debug.Log(center);
            trackedObject.transform.position = center;
            trackedObject.transform.localScale = new Vector3(quadWidth, quadHeight, 1f);
        }
    }
}