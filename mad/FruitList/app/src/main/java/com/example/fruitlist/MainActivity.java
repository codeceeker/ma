package com.example.fruitlist;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ListView;

public class MainActivity extends AppCompatActivity {

    String fruitList[]={"Apple","Banana","Apricot","Orange","Water Melon"};
    int fruitImages[]={R.drawable.apple,R.drawable.banana,R.drawable.apricot,R.drawable.orange,R.drawable.watermelon};



    ListView listView;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        listView =(ListView)findViewById(R.id.customListView);
        CustomBaseAdapter customBaseAdapter=new CustomBaseAdapter(getApplicationContext(),fruitList,fruitImages);
        listView.setAdapter(customBaseAdapter);
        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                Log.i("CUSTOM_LIST_VIEW", "Item is clicked @position::"+position);
            }
        });

    }
}